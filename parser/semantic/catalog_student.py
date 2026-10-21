"""Validate and prompt student pairs for the versioned S1 document target."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybrid_miner.contracts.semantic_document import SemanticDocument

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.catalog_document_parser import (
    prepare_catalog_target,
)
from parser.semantic.catalog_prompt import build_catalog_document_prompt

_PAIR_FIELDS = frozenset(
    {"contract_version", "document_id", "corpus_id", "input", "parser", "label"}
)
_SOURCE_METADATA = frozenset({"source_dataset", "source_domain"})


def document_target(document: SemanticDocument) -> dict[str, Any]:
    """Recover the same quote-based target from a validated document.

    Only representation changes: references, polarity, factuality, rule scope,
    and hypotheses are preserved. Runtime provenance is not a model label.
    """

    def anchor(span):
        text = document.source_text[span.start : span.end]
        positions = []
        start = 0
        while (position := document.source_text.find(text, start)) != -1:
            positions.append(position)
            start = position + 1
        return {"text": text, "occurrence": positions.index(span.start)}

    def quoted(value):
        result = value.model_dump(mode="json")
        result.pop("source_spans")
        result["source_quotes"] = [anchor(span) for span in value.source_spans]
        return result

    rules = []
    for rule in document.rules:
        target = quoted(rule)
        target["antecedents"] = [quoted(a) for a in rule.antecedents]
        target["consequents"] = [quoted(a) for a in rule.consequents]
        rules.append(target)
    return {
        "entities": [
            {
                "id": e.id,
                "type": e.type,
                "mention": anchor(e.span)["text"] if e.span is not None else None,
                "occurrence": (
                    anchor(e.span)["occurrence"] if e.span is not None else None
                ),
            }
            for e in document.entities
        ],
        "assertions": [quoted(a) for a in document.assertions],
        "rules": rules,
    }


def make_student_pair(
    document: SemanticDocument,
    *,
    target: dict[str, Any] | None = None,
    source_dataset: str | None = None,
    source_domain: str | None = None,
) -> dict[str, Any]:
    """Encode one document as a supervised pair, not a second semantic target."""
    pair = {
        "contract_version": document.contract_version,
        "document_id": document.document_id,
        "corpus_id": document.corpus_id,
        "input": document.source_text,
        "parser": document.parser.model_dump(mode="json"),
        "label": json.dumps(
            target if target is not None else document_target(document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for field, value in (
        ("source_dataset", source_dataset),
        ("source_domain", source_domain),
    ):
        if value is not None:
            pair[field] = value
    return pair


def validate_catalog_student_pair(
    pair: object, *, catalog: OntologyCatalog
) -> tuple[str, str, SemanticDocument]:
    """Check the label against the same contract used before teacher export.

    The full prompt is regenerated from the exact source and ontology versions.
    A pair from an older prompt or bundle cannot silently enter a new run.
    """

    if (
        not isinstance(pair, dict)
        or not set(pair) >= _PAIR_FIELDS
        or set(pair) - _PAIR_FIELDS - _SOURCE_METADATA
    ):
        raise ValueError("Catalog student pair has missing or unexpected fields")
    for field in _SOURCE_METADATA & pair.keys():
        if not isinstance(pair[field], str) or not pair[field].strip():
            raise ValueError(f"{field} must be nonempty text")
    if pair["contract_version"] != "semantic-document-v1":
        raise ValueError("Unsupported catalog student contract version")
    source_text = pair["input"]
    label = pair["label"]
    provenance = pair["parser"]
    if not isinstance(source_text, str) or not isinstance(label, str):
        raise ValueError("Catalog student input and label must be strings")
    if not isinstance(provenance, dict):
        raise ValueError("Catalog student parser provenance must be an object")
    prompt = build_catalog_document_prompt(source_text, catalog=catalog)
    if provenance.get("prompt_version") != prompt.version:
        raise ValueError(
            "Catalog student prompt version does not match the active prompt"
        )
    if provenance.get("ontology_versions") != list(catalog.versions):
        raise ValueError("Catalog student ontology versions do not match the bundle")
    try:
        payload: Any = json.loads(label)
    except json.JSONDecodeError as error:
        raise ValueError("Catalog student label is not JSON") from error
    if not isinstance(payload, dict) or set(payload) not in (
        {"entities", "assertions"},
        {"entities", "assertions", "rules"},
    ):
        raise ValueError(
            "Catalog student label must contain entities, assertions and optional rules"
        )
    document, payload = prepare_catalog_target(
        payload,
        document_id=pair["document_id"],
        corpus_id=pair["corpus_id"],
        source_text=source_text,
        parser_provenance=provenance,
        catalog=catalog,
    )
    return (
        prompt.text,
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        document,
    )


def validate_catalog_student_checkpoint(
    model_directory: str | Path, *, catalog: OntologyCatalog
) -> None:
    """Reject a student checkpoint trained for another S1 target or bundle."""

    path = Path(model_directory) / "training_metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read student training metadata at {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("Student training metadata must be an object")
    prompt_version = build_catalog_document_prompt(
        "Checkpoint compatibility.", catalog=catalog
    ).version
    if metadata.get("target_contract_version") != "semantic-document-v1":
        raise ValueError("Student checkpoint has the wrong target contract")
    if metadata.get("prompt_version") != prompt_version:
        raise ValueError("Student checkpoint has the wrong prompt version")
    if metadata.get("ontology_versions") != list(catalog.versions):
        raise ValueError("Student checkpoint has the wrong ontology versions")
