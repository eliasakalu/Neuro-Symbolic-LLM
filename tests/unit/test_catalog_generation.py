"""Tests for the untrusted catalog-guided model-generation step."""

import json

import pytest

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)


class RecordingBackend:
    provider_name = "test"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return self.response


def test_all_generation_uses_the_document_prompt() -> None:
    prompt = ReferenceSemanticParser.build_prompt("Alice opened the door.")
    assert SemanticParserConfig(model_name="teacher-model").prompt_version == "2.0.0"
    assert "<worked_example>" in prompt
    assert "Shared source interpretation rules:" in prompt


def test_instance_prompt_and_generation_use_the_selected_catalog():
    catalog = load_ontology_catalog(modules=("upper",))
    backend = RecordingBackend('{"entities": [], "assertions": [], "rules": []}')
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="review"),
        catalog=catalog,
    )
    inspected = parser.build_document_prompt("Mira left.")
    parser.parse("Mira left.")
    assert inspected.text == backend.calls[0][0]
    assert "Ontology versions: upper-v1\n" in inspected.text


def test_reference_context_is_not_added_to_the_extraction_source():
    from tests.semantic_fixtures import target

    backend = RecordingBackend(json.dumps(target("She left.")))
    parser = ReferenceSemanticParser(
        backend=backend, config=SemanticParserConfig(model_name="review")
    )
    document = parser.parse("She left.", context="Nora arrived yesterday.")
    assert document.source_text == "She left."
    assert (
        "<reference_context>\nNora arrived yesterday.\n</reference_context>"
        in backend.calls[0][0]
    )
    assert document.assertions[0].source_spans[0].end == len("She left.")
    backend.response = json.dumps(target("Nora arrived yesterday."))
    with pytest.raises(SemanticParseError, match="absent from source"):
        parser.parse("She left.", context="Nora arrived yesterday.")


def test_uses_existing_backend_and_retains_prompt_version() -> None:
    backend = RecordingBackend('```json\n{"entities": [], "assertions": []}\n```')
    source = "  Mira left.  "

    parser = ReferenceSemanticParser(
        catalog=load_ontology_catalog(),
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
    )
    payload, prompt_version = parser.generate_document_target(source)

    assert (
        prompt_version
        == build_catalog_document_prompt(
            source, catalog=load_ontology_catalog()
        ).version
    )
    assert payload == {"entities": [], "assertions": []}
    assert backend.calls[0][1] == "teacher-model"
    assert f"<source_text>\n{source}\n</source_text>" in backend.calls[0][0]


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        json.dumps({"entities": [], "assertions": [], "source_text": "forged"}),
        json.dumps({"entities": {}, "assertions": []}),
    ],
)
def test_rejects_wrong_envelope_and_model_supplied_metadata(response: str) -> None:
    with pytest.raises(SemanticParseError, match="Document output|must be lists"):
        ReferenceSemanticParser(
            catalog=load_ontology_catalog(),
            backend=RecordingBackend(response),
            config=SemanticParserConfig(model_name="teacher-model"),
        ).generate_document_target("Mira left.")


def test_rejects_invalid_or_empty_model_output() -> None:
    for response, error_type in (
        ("not JSON", SemanticParseError),
        ("  ", SemanticParseError),
    ):
        with pytest.raises(error_type):
            ReferenceSemanticParser(
                catalog=load_ontology_catalog(),
                backend=RecordingBackend(response),
                config=SemanticParserConfig(model_name="teacher-model"),
            ).generate_document_target("Mira left.")
