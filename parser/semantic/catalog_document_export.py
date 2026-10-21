"""Export validated catalog semantic documents from source-record JSONL."""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from parser.semantic.semantic_parser import ReferenceSemanticParser

if TYPE_CHECKING:
    from hybrid_miner.ontology.bundle import OntologyBundle

_SOURCE_FIELDS = frozenset({"document_id", "corpus_id", "source_text"})
_IDENTIFIER = re.compile(r"^[^\s]+$")


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Counts and paths for one completed document export."""

    accepted: int
    rejected: int
    output_path: Path
    rejected_path: Path
    student_pairs_path: Path | None = None
    bundle_path: Path | None = None


def _source_fields(raw_line: str) -> tuple[str, str, str]:
    if not raw_line.strip():
        raise ValueError("Source JSONL line cannot be empty")
    value = json.loads(raw_line)
    if not isinstance(value, dict) or set(value) != _SOURCE_FIELDS:
        raise ValueError(
            "Source record requires exactly document_id, corpus_id, and source_text"
        )
    document_id = value["document_id"]
    corpus_id = value["corpus_id"]
    source_text = value["source_text"]
    for name, identifier in (("document_id", document_id), ("corpus_id", corpus_id)):
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(f"{name} must be a nonempty identifier without whitespace")
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError("source_text must contain non-whitespace text")
    return document_id, corpus_id, source_text


def source_record(
    raw_line: str,
    *,
    jsonl: bool,
    corpus_id: str,
) -> tuple[str, str, str, dict[str, str]]:
    """Reuse sentence and pilot-manifest inputs without losing their metadata."""
    if jsonl:
        value = json.loads(raw_line)
        if isinstance(value, dict) and set(value) == _SOURCE_FIELDS:
            return (*_source_fields(raw_line), {})
        if (
            not isinstance(value, dict)
            or "text" not in value
            or set(value) - {"text", "source_dataset", "source_domain"}
        ):
            raise ValueError(
                "Expected source IDs/text or the text/dataset/domain manifest"
            )
        text = value["text"]
        metadata = {}
        for field in ("source_dataset", "source_domain"):
            item = value.get(field)
            if item is not None:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"{field} must be nonempty text or null")
                metadata[field] = item
        namespace = metadata.get("source_dataset", corpus_id)
    else:
        text, metadata, namespace = raw_line, {}, corpus_id
    if not isinstance(text, str) or not text.strip():
        raise ValueError("source_text must contain non-whitespace text")
    if not _IDENTIFIER.fullmatch(namespace):
        raise ValueError(
            "Corpus/dataset ID must not contain whitespace; supply --corpus-id"
        )
    identifier = "text-" + sha256(text.encode("utf-8")).hexdigest()
    return identifier, namespace, text, metadata


def export_catalog_documents(
    parser: ReferenceSemanticParser,
    *,
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path,
    student_pairs_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    ontology_bundle: OntologyBundle | None = None,
    corpus_id: str | None = None,
) -> ExportResult:
    """Export validated documents and optional labels from one model call each."""

    source = Path(input_path)
    accepted = Path(output_path)
    rejected = Path(rejected_path)
    pairs = Path(student_pairs_path) if student_pairs_path is not None else None
    bundle_output = Path(bundle_path) if bundle_path is not None else None
    if bundle_output is not None and ontology_bundle is None:
        raise ValueError("Bundle output requires the loaded ontology bundle")
    outputs: tuple[Path, ...] = (accepted, rejected)
    if pairs is not None:
        outputs += (pairs,)
    if bundle_output is not None:
        outputs += (bundle_output,)
    paths = (source, *outputs)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError(
            "Input, accepted-output, rejected-output, student-pairs, and bundle paths must all differ"
        )
    if not source.is_file():
        raise FileNotFoundError(f"Source JSONL file not found: {source}")
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    accepted.parent.mkdir(parents=True, exist_ok=True)
    rejected.parent.mkdir(parents=True, exist_ok=True)
    if pairs is not None:
        pairs.parent.mkdir(parents=True, exist_ok=True)
    if bundle_output is not None:
        bundle_output.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
    rejected_count = 0
    seen_documents: set[tuple[str, str]] = set()
    created_paths: list[Path] = []
    try:
        with ExitStack() as files:
            source_file = files.enter_context(source.open("r", encoding="utf-8"))
            accepted_file = files.enter_context(
                accepted.open("x", encoding="utf-8", newline="\n")
            )
            created_paths.append(accepted)
            rejected_file = files.enter_context(
                rejected.open("x", encoding="utf-8", newline="\n")
            )
            created_paths.append(rejected)
            pairs_file = None
            if bundle_output is not None:
                bundle_file = files.enter_context(
                    bundle_output.open("x", encoding="utf-8", newline="\n")
                )
                created_paths.append(bundle_output)
                assert ontology_bundle is not None
                bundle_file.write(ontology_bundle.canonical_json() + "\n")
            if pairs is not None:
                pairs_file = files.enter_context(
                    pairs.open("x", encoding="utf-8", newline="\n")
                )
                created_paths.append(pairs)
            for line_number, line in enumerate(source_file, start=1):
                raw_line = line.removesuffix("\n").removesuffix("\r")
                try:
                    document_id, record_corpus, source_text, source_metadata = (
                        source_record(
                            raw_line,
                            jsonl=source.suffix.lower() == ".jsonl",
                            corpus_id=(
                                corpus_id if corpus_id is not None else source.stem
                            ),
                        )
                    )
                    identity = (record_corpus, document_id)
                    if identity in seen_documents:
                        raise ValueError(f"Duplicate source document {identity!r}")
                    seen_documents.add(identity)
                    if pairs_file is None:
                        document = parser.parse_document(
                            document_id=document_id,
                            corpus_id=record_corpus,
                            source_text=source_text,
                        )
                    else:
                        document, target = parser.parse_document_with_target(
                            document_id=document_id,
                            corpus_id=record_corpus,
                            source_text=source_text,
                        )
                except ValueError as error:
                    rejected_file.write(
                        json.dumps(
                            {
                                "line_number": line_number,
                                "raw_line": raw_line,
                                "error": str(error),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rejected_count += 1
                    continue
                accepted_file.write(document.canonical_json() + "\n")
                if pairs_file is not None:
                    from parser.semantic.catalog_student import make_student_pair

                    pairs_file.write(
                        json.dumps(
                            make_student_pair(
                                document, target=target, **source_metadata
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                accepted_count += 1
    except BaseException:
        # Only remove paths created by this invocation. An interrupted run must
        # not leave partial files that appear to be completed exports.
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    return ExportResult(
        accepted=accepted_count,
        rejected=rejected_count,
        output_path=accepted,
        rejected_path=rejected,
        student_pairs_path=pairs,
        bundle_path=bundle_output,
    )
