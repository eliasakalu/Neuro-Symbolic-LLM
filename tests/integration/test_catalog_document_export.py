"""Integration tests for the parser-to-miner JSONL export boundary."""

# ruff: noqa: E402, I001

import json
from pathlib import Path

import pytest

pytest.importorskip("hybrid_miner.contracts.semantic_document")

from hybrid_miner.contracts.semantic_document import SemanticDocument  # noqa: E402

from parser.ontology.catalog import load_ontology_catalog  # noqa: E402
from parser.ontology.miner_bundle import load_catalog_bundle  # noqa: E402
from parser.semantic import catalog_document_cli  # noqa: E402
from parser.semantic import cli, model_config  # noqa: E402
from parser.semantic.backends import CallableBackend  # noqa: E402
from parser.semantic.catalog_document_export import (
    export_catalog_documents,
)  # noqa: E402
from parser.semantic.catalog_document_parser import (  # noqa: E402
    CatalogSemanticDocumentParser,
)
from parser.semantic.semantic_parser import ModelGenerationError  # noqa: E402


def _source_line(
    document_id: str, source_text: str, corpus_id: str = "corpus-1"
) -> str:
    return json.dumps(
        {
            "document_id": document_id,
            "corpus_id": corpus_id,
            "source_text": source_text,
        },
        ensure_ascii=False,
    )


def test_exports_canonical_documents_and_auditable_rejections(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    student_pairs_path = tmp_path / "student-pairs.jsonl"
    source_text = "  Mira left. ðŸ™‚  "
    input_path.write_text(
        "\n".join(
            [
                _source_line("doc-1", source_text),
                "not-json",
                _source_line("doc-1", "A duplicate."),
                _source_line("bad id", "An invalid identifier."),
                _source_line("doc-2", "Sol arrived."),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def generate(*, prompt: str, model: str) -> str:
        calls.append(prompt)
        return '{"entities": [], "assertions": []}'

    parser = CatalogSemanticDocumentParser(
        backend=CallableBackend(generate, provider_name="test-provider"),
        model_name="teacher-model",
        catalog=load_ontology_catalog(modules=("upper",)),
    )
    result = export_catalog_documents(
        parser,
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        student_pairs_path=student_pairs_path,
    )

    assert (result.accepted, result.rejected) == (2, 3)
    assert len(calls) == 2
    accepted_lines = output_path.read_text(encoding="utf-8").splitlines()
    documents = [SemanticDocument.model_validate_json(line) for line in accepted_lines]
    assert [document.document_id for document in documents] == ["doc-1", "doc-2"]
    assert documents[0].source_text == source_text
    assert documents[0].parser.ontology_versions == ("upper-v1",)
    assert accepted_lines == [document.canonical_json() for document in documents]
    rejections = [
        json.loads(line)
        for line in rejected_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [rejection["line_number"] for rejection in rejections] == [2, 3, 4]
    assert "Duplicate source document" in rejections[1]["error"]
    assert rejections[0]["raw_line"] == "not-json"
    pairs = [
        json.loads(line)
        for line in student_pairs_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [pair["document_id"] for pair in pairs] == ["doc-1", "doc-2"]
    assert pairs[0]["contract_version"] == "semantic-document-v1"
    assert json.loads(pairs[0]["label"]) == {"entities": [], "assertions": []}
    assert pairs[0]["parser"] == documents[0].parser.model_dump(mode="json")


def test_rejects_colliding_paths_and_existing_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jsonl"
    input_path.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    parser = CatalogSemanticDocumentParser(
        backend=CallableBackend(lambda **kwargs: '{"entities": [], "assertions": []}'),
        model_name="teacher-model",
    )
    rejected_path = tmp_path / "rejected.jsonl"

    with pytest.raises(ValueError, match="must all differ"):
        export_catalog_documents(
            parser,
            input_path=input_path,
            output_path=input_path,
            rejected_path=rejected_path,
        )
    assert input_path.read_text(encoding="utf-8").startswith('{"document_id"')

    output_path = tmp_path / "parsed.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Output already exists"):
        export_catalog_documents(
            parser,
            input_path=input_path,
            output_path=output_path,
            rejected_path=rejected_path,
        )
    assert output_path.read_text(encoding="utf-8") == "existing\n"


def test_backend_failure_stops_export_without_partial_output(tmp_path: Path) -> None:
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    input_path.write_text(
        _source_line("doc-1", "Mira left.")
        + "\n"
        + _source_line("doc-2", "Sol arrived.")
        + "\n",
        encoding="utf-8",
    )

    def generate(*, prompt: str, model: str) -> str:
        if "Sol arrived." in prompt:
            raise ConnectionError("model server unavailable")
        return '{"entities": [], "assertions": []}'

    parser = CatalogSemanticDocumentParser(
        backend=CallableBackend(generate), model_name="teacher-model"
    )

    with pytest.raises(ModelGenerationError, match="parser failed"):
        export_catalog_documents(
            parser,
            input_path=input_path,
            output_path=output_path,
            rejected_path=rejected_path,
        )

    assert not output_path.exists()
    assert not rejected_path.exists()
    assert input_path.read_text(encoding="utf-8").count("\n") == 2


def test_cli_uses_selected_catalog_and_reports_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "parsed.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    input_path.write_text(
        _source_line("doc-1", "Mira left.") + "\n" + "not-json\n",
        encoding="utf-8",
    )
    backend = CallableBackend(lambda **kwargs: '{"entities": [], "assertions": []}')
    monkeypatch.setattr(model_config, "create_backend", lambda profile: backend)

    exit_code = catalog_document_cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--rejected-output",
            str(rejected_path),
            "--ontology-module",
            "upper",
        ]
    )

    assert exit_code == 2
    document = SemanticDocument.model_validate_json(
        output_path.read_text(encoding="utf-8").strip()
    )
    assert document.parser.ontology_versions == ("upper-v1",)
    assert len(rejected_path.read_text(encoding="utf-8").splitlines()) == 1


def test_existing_cli_exports_documents_with_the_selected_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    selected = []

    def backend(profile):
        selected.append(profile.name)
        return CallableBackend(lambda **kwargs: '{"entities": [], "assertions": []}')

    monkeypatch.setattr(model_config, "create_backend", backend)
    status = cli.main(
        [
            "--profile",
            "ollama_catalog",
            "export-documents",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "documents.jsonl"),
            "--rejected-output",
            str(tmp_path / "rejected.jsonl"),
            "--student-pairs-output",
            str(tmp_path / "pairs.jsonl"),
        ]
    )
    assert status == 0
    assert selected == ["ollama_catalog"]
    assert (tmp_path / "pairs.jsonl").read_text(encoding="utf-8")


def test_export_includes_the_same_bundle_used_to_validate_documents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    catalog, bundle = load_catalog_bundle()
    parser = CatalogSemanticDocumentParser(
        backend=CallableBackend(lambda **kwargs: '{"entities": [], "assertions": []}'),
        model_name="teacher-model",
        catalog=catalog,
    )
    result = export_catalog_documents(
        parser,
        input_path=source,
        output_path=tmp_path / "documents.jsonl",
        rejected_path=tmp_path / "rejected.jsonl",
        bundle_path=tmp_path / "bundle.json",
        ontology_bundle=bundle,
    )
    assert result.bundle_path is not None
    assert (
        result.bundle_path.read_text(encoding="utf-8") == bundle.canonical_json() + "\n"
    )


def test_build_dataset_document_target_matches_export_and_student_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from parser.semantic.catalog_student import validate_catalog_student_pair

    source = tmp_path / "source.jsonl"
    source.write_text(_source_line("doc-1", "Mira left.") + "\n", encoding="utf-8")
    draft = {
        "entities": [
            {"id": "event", "type": " Event ", "mention": "left", "occurrence": 0}
        ],
        "assertions": [
            {
                "predicate": " Event ",
                "arguments": ["event"],
                "argument_types": [" Event "],
                "status": "extracted-hypothesis",
                "factuality": "asserted",
                "confidence": 0.9,
                "source_quotes": [{"text": "left", "occurrence": 0}],
                "alternatives": [],
            }
        ],
    }
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return json.dumps(draft)

    monkeypatch.setattr(
        model_config,
        "create_backend",
        lambda profile: CallableBackend(generate),
    )
    pairs = tmp_path / "pairs.jsonl"
    documents = tmp_path / "documents.jsonl"
    status = cli.main(
        [
            "build-dataset",
            "--target",
            "semantic-document-v1",
            "--input",
            str(source),
            "--output",
            str(documents),
            "--student-pairs-output",
            str(pairs),
            "--rejected-output",
            str(tmp_path / "rejected.jsonl"),
            "--ontology-module",
            "upper",
        ]
    )
    assert status == 0
    assert len(calls) == 1
    pair = json.loads(pairs.read_text(encoding="utf-8"))
    _, label, rebuilt = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog(modules=("upper",))
    )
    assert rebuilt.canonical_json() + "\n" == documents.read_text(encoding="utf-8")
    assert json.loads(label) == json.loads(pair["label"])
    assert json.loads(label)["assertions"][0]["conditional_on"] is None
    assert json.loads(label)["assertions"][0]["predicate"] == "Event"


def test_document_dataset_rejects_lossy_filter_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def unexpected(profile):
        pytest.fail("Backend must not be constructed for incompatible options")

    monkeypatch.setattr(model_config, "create_backend", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "build-dataset",
                "--input",
                str(tmp_path / "source.jsonl"),
                "--output",
                str(tmp_path / "documents.jsonl"),
                "--rejected-output",
                str(tmp_path / "rejected.jsonl"),
                "--asserted-only",
            ]
        )
    assert error.value.code == 2
