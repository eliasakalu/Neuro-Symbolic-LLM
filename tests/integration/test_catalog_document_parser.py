"""Cross-repository tests for catalog-guided semantic-document validation."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("hybrid_miner.contracts.semantic_document")

from hybrid_miner.contracts.semantic_document import SemanticDocument  # noqa: E402

from parser.ontology.catalog import load_ontology_catalog  # noqa: E402
from parser.semantic.catalog_document_parser import (  # noqa: E402
    CATALOG_PARSER_VERSION,
    CatalogSemanticDocumentParser,
)
from parser.semantic.catalog_prompt import build_catalog_document_prompt  # noqa: E402
from parser.semantic.semantic_parser import (  # noqa: E402
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)


class RecordingBackend:
    provider_name = "test-provider"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, prompt: str, model: str) -> str:
        self.calls.append((prompt, model))
        return json.dumps(self.payload)


def _source_and_payload() -> tuple[str, dict[str, Any]]:
    source = "🙂 Although Mira promised Sol, she delayed."
    clause_end = source.index(",")
    second_start = source.index("she")
    return source, {
        "entities": [
            {
                "id": "clause-a",
                "type": "Clause",
                "mention": source[2:clause_end],
                "occurrence": 0,
            },
            {
                "id": "clause-b",
                "type": "Clause",
                "mention": source[second_start:],
                "occurrence": 0,
            },
        ],
        "assertions": [
            {
                "predicate": "Concession",
                "arguments": ["clause-a", "clause-b"],
                "argument_types": ["Clause", "Clause"],
                "status": "extracted-hypothesis",
                "factuality": "asserted",
                "confidence": 0.9,
                "source_quotes": [{"text": source[2:], "occurrence": 0}],
                "alternatives": [],
            }
        ],
    }


def _parse(payload: dict[str, Any], source: str) -> SemanticDocument:
    parser = CatalogSemanticDocumentParser(
        backend=RecordingBackend(payload), model_name="teacher-model"
    )
    return parser.parse_document(
        document_id="doc-1", corpus_id="corpus-1", source_text=source
    )


def test_returns_validated_document_with_runtime_provenance() -> None:
    source, payload = _source_and_payload()
    backend = RecordingBackend(payload)
    parser = CatalogSemanticDocumentParser(
        backend=backend,
        model_name="teacher-model",
        model_revision="revision-1",
    )

    document = parser.parse_document(
        document_id="doc-1", corpus_id="corpus-1", source_text=source
    )

    assert isinstance(document, SemanticDocument)
    assert document.source_text == source
    assert document.parser.parser_version == CATALOG_PARSER_VERSION
    assert (
        document.parser.prompt_version
        == build_catalog_document_prompt(
            source, catalog=load_ontology_catalog()
        ).version
    )
    assert document.parser.ontology_versions == load_ontology_catalog().versions
    assert document.parser.provider == "test-provider"
    assert document.parser.model_revision == "revision-1"
    assert document.assertions[0].source_spans[0].end == len(source)
    assert f"<source_text>\n{source}\n</source_text>" in backend.calls[0][0]
    assert (
        document.canonical_json()
        == SemanticDocument.model_validate_json(
            document.canonical_json()
        ).canonical_json()
    )


def test_event_type_assertion_and_agent_share_the_same_typed_event() -> None:
    source = "Mira left."
    payload = {
        "entities": [
            {"id": "mira", "type": "Entity", "mention": "Mira", "occurrence": 0},
            {"id": "departure", "type": "Event", "mention": "left", "occurrence": 0},
        ],
        "assertions": [
            {
                "predicate": "Event",
                "arguments": ["departure"],
                "argument_types": ["Event"],
                "status": "extracted-hypothesis",
                "factuality": "asserted",
                "confidence": 0.95,
                "source_quotes": [{"text": "left", "occurrence": 0}],
                "alternatives": [],
            },
            {
                "predicate": "Agent",
                "arguments": ["departure", "mira"],
                "argument_types": ["Event", "Entity"],
                "status": "extracted-hypothesis",
                "factuality": "asserted",
                "confidence": 0.9,
                "source_quotes": [{"text": "Mira left", "occurrence": 0}],
                "alternatives": [],
            },
        ],
    }

    document = _parse(payload, source)

    assert document.assertions[0].arguments == ("departure",)
    assert document.assertions[1].arguments == ("departure", "mira")


def test_worked_examples_are_valid_miner_documents() -> None:
    examples_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "parser_config"
        / "catalog_document_examples.jsonl"
    )
    catalog = load_ontology_catalog()
    examples = examples_path.read_text(encoding="utf-8").splitlines()

    assert examples, "The prompt must contain worked examples"
    covered = {
        version
        for line in examples
        for version in json.loads(line)["required_ontology_versions"]
    }
    assert covered == set(catalog.versions)
    for index, line in enumerate(examples, start=1):
        example = json.loads(line)
        parser = CatalogSemanticDocumentParser(
            backend=RecordingBackend(
                {
                    "entities": example["entities"],
                    "assertions": example["assertions"],
                    "rules": example.get("rules", []),
                }
            ),
            model_name="curated-example",
            catalog=catalog,
        )
        document = parser.parse_document(
            document_id=f"example-{index}",
            corpus_id="catalog-examples",
            source_text=example["source_text"],
        )

        assert document.assertions or document.rules
        assert (
            document.parser.prompt_version
            == build_catalog_document_prompt(
                example["source_text"], catalog=catalog
            ).version
        )


def test_rejects_unknown_predicate_without_provisional_status() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["predicate"] = "InventedRelation"

    with pytest.raises(SemanticParseError, match="unknown predicate"):
        _parse(payload, source)


def test_accepts_explicit_provisional_relation() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["predicate"] = "ProvisionalRelation"
    payload["assertions"][0]["provisional_relation_id"] = "unclassified_contrast"

    document = _parse(payload, source)

    assert document.assertions[0].provisional_relation_id == "unclassified_contrast"


def test_rejects_confidence_outside_catalog_range() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["confidence"] = 0.3

    with pytest.raises(SemanticParseError, match="outside catalog range"):
        _parse(payload, source)


def test_rejects_undeclared_entity_type() -> None:
    source, payload = _source_and_payload()
    payload["entities"][0]["type"] = "Artifact"
    payload["assertions"] = []

    with pytest.raises(SemanticParseError, match="Undeclared ontology entity type"):
        _parse(payload, source)


def test_rejects_evidence_not_present_in_source() -> None:
    source, payload = _source_and_payload()
    payload["assertions"][0]["source_quotes"][0]["text"] = "Not in source"

    with pytest.raises(SemanticParseError, match="is absent from source text"):
        _parse(payload, source)


def test_preserves_conditional_factuality_without_claiming_derivation() -> None:
    source = "If it rains, Mira may leave."
    payload = {
        "entities": [
            {
                "id": "rain-condition",
                "type": "Clause",
                "mention": "it rains",
                "occurrence": 0,
            },
            {
                "id": "departure",
                "type": "Event",
                "mention": "leave",
                "occurrence": 0,
            },
        ],
        "assertions": [
            {
                "predicate": "Event",
                "arguments": ["departure"],
                "argument_types": ["Event"],
                "status": "extracted-hypothesis",
                "factuality": "hypothetical",
                "conditional_on": "rain-condition",
                "confidence": 0.95,
                "source_quotes": [{"text": "Mira may leave", "occurrence": 0}],
                "alternatives": [],
            }
        ],
    }

    document = _parse(payload, source)

    assert document.assertions[0].status == "extracted-hypothesis"
    assert document.assertions[0].factuality == "hypothetical"
    assert document.assertions[0].conditional_on == "rain-condition"
    assert document.assertions[0].source_spans[0].start == source.index("Mira")


def test_rejects_missing_factuality_in_catalog_output() -> None:
    source, payload = _source_and_payload()
    del payload["assertions"][0]["factuality"]

    with pytest.raises(SemanticParseError, match="explicit factuality"):
        _parse(payload, source)


def test_shared_parser_and_compatibility_constructor_use_identical_target() -> None:
    source, payload = _source_and_payload()
    catalog = load_ontology_catalog()
    backend = RecordingBackend(payload)
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        catalog=catalog,
    )
    document, target = parser.parse_document_with_target(
        document_id="shared-doc", corpus_id="shared-corpus", source_text=source
    )
    compatibility = CatalogSemanticDocumentParser(
        backend=RecordingBackend(payload), model_name="teacher-model", catalog=catalog
    ).parse_document(
        document_id="shared-doc", corpus_id="shared-corpus", source_text=source
    )
    assert document == compatibility
    from parser.semantic.normalization import normalize_document_target

    assert target == normalize_document_target(payload)
    assert "conditional_on" not in payload["assertions"][0]
    assert len(backend.calls) == 1
    assert "parse_document" not in CatalogSemanticDocumentParser.__dict__


def test_document_target_preserves_an_empty_rule_envelope() -> None:
    parser = ReferenceSemanticParser(
        backend=RecordingBackend({"entities": [], "assertions": [], "rules": []}),
        config=SemanticParserConfig(model_name="teacher-model"),
    )
    document = parser.parse_document(
        document_id="rule-doc", corpus_id="rule-corpus", source_text="If it rains."
    )
    assert document.rules == ()


def test_required_schema_decoding_rejects_an_unsupported_backend_before_generation() -> (
    None
):
    backend = RecordingBackend({"entities": [], "assertions": []})
    parser = ReferenceSemanticParser(
        backend=backend,
        config=SemanticParserConfig(model_name="teacher-model"),
        require_structured_output=True,
    )
    with pytest.raises(ValueError, match="does not support JSON-schema decoding"):
        parser.parse_document(
            document_id="doc", corpus_id="corpus", source_text="Mira left."
        )
    assert not backend.calls


def test_decoding_schema_reuses_the_final_contract_with_quote_anchors() -> None:
    from parser.semantic.catalog_output_schema import catalog_output_schema

    catalog = load_ontology_catalog()
    schema = catalog_output_schema(catalog)
    branches = schema["$defs"]["SemanticAssertion"]["anyOf"]
    assertion = next(
        branch
        for branch in branches
        if branch["properties"]["predicate"]["const"] == "ProvisionalRelation"
    )
    entity = schema["$defs"]["EntityRecord"]
    original = SemanticDocument.model_json_schema()["$defs"]["SemanticAssertion"]
    assert assertion["properties"]["confidence"] == original["properties"]["confidence"]
    assert "source_spans" not in assertion["properties"]
    assert "source_quotes" in assertion["required"]
    assert "conditional_on" in assertion["required"]
    assert "span" not in entity["properties"]
    assert set(entity["required"]) == {"id", "type", "mention", "occurrence"}
    assert set(schema["properties"]) == {"entities", "assertions", "rules"}
    assert "rules" in schema["required"]
    assert schema["$defs"]["SemanticRule"]["properties"]["variables"]["items"] == {
        "$ref": "#/$defs/RuleVariable"
    }


def test_decoding_branches_follow_every_active_catalog_signature() -> None:
    from parser.semantic.catalog_output_schema import catalog_output_schema

    catalog = load_ontology_catalog()
    branches = {
        branch["properties"]["predicate"]["const"]: branch
        for branch in catalog_output_schema(catalog)["$defs"]["SemanticAssertion"][
            "anyOf"
        ]
    }
    assert set(branches) == {*catalog.relations, "ProvisionalRelation"}
    for name, relation in catalog.relations.items():
        assert "conditional_on" in branches[name]["required"]
        properties = branches[name]["properties"]
        assert properties["arguments"]["minItems"] == relation.arity
        assert properties["arguments"]["maxItems"] == relation.arity
        assert properties["argument_types"] == {
            "type": "array",
            "prefixItems": [
                {"type": "string", "enum": list(argument.allowed_types)}
                for argument in relation.arguments
            ],
            "minItems": relation.arity,
            "maxItems": relation.arity,
        }
        assert properties["confidence"]["minimum"] == relation.confidence_range[0]
        assert properties["confidence"]["maximum"] == relation.confidence_range[1]
        assert properties["provisional_relation_id"] == {"type": "null"}
    provisional = branches["ProvisionalRelation"]
    assert "provisional_relation_id" in provisional["required"]
    assert provisional["properties"]["provisional_relation_id"]["type"] == "string"


def test_decoding_schema_does_not_expose_inactive_catalogs() -> None:
    from parser.semantic.catalog_output_schema import catalog_output_schema

    schema = catalog_output_schema(load_ontology_catalog(modules=("upper",)))
    names = {
        branch["properties"]["predicate"]["const"]
        for branch in schema["$defs"]["SemanticAssertion"]["anyOf"]
    }
    assert "Agent" in names
    assert "Concession" not in names


def test_decoding_scope_is_explicit_and_restricted_to_hypothetical_assertions() -> None:
    from parser.semantic.catalog_output_schema import catalog_output_schema

    catalog = load_ontology_catalog()
    branches = catalog_output_schema(catalog)["$defs"]["SemanticAssertion"]["anyOf"]
    assert len(branches) == 2 * (len(catalog.relations) + 1)
    for branch in branches:
        properties = branch["properties"]
        assert "conditional_on" in branch["required"]
        if properties["factuality"].get("const") == "hypothetical":
            assert properties["conditional_on"] == {
                "anyOf": [
                    {"type": "string", "minLength": 1, "pattern": r"^[^\s]+$"},
                    {"type": "null"},
                ]
            }
        else:
            assert properties["factuality"]["enum"] == [
                "asserted",
                "opinion",
                "speculative",
            ]
            assert properties["conditional_on"] == {"type": "null"}
