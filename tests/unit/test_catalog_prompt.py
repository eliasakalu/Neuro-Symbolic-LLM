"""Tests for catalog-guided semantic-document prompt construction."""

import json
import re
from dataclasses import replace

import jsonschema
import pytest
import yaml

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic import catalog_prompt
from parser.semantic.backends import CallableBackend
from parser.semantic.catalog_document_parser import prepare_catalog_target
from parser.semantic.catalog_output_schema import catalog_output_schema
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParserConfig,
)


def test_prompt_uses_catalog_signatures_and_versions() -> None:
    prompt = build_catalog_document_prompt(
        "Although it rained, Mira left.", catalog=load_ontology_catalog()
    )

    assert prompt.version.startswith("catalog-document-v13+sha256.")
    assert len(prompt.version.split("+sha256.")[1]) == 64
    assert "discourse-v1" in prompt.text
    assert "upper-v1" in prompt.text
    assert "Concession(statement: Clause, counter_statement: Clause)" in prompt.text
    assert "Causes(cause: Event|Proposition, effect: Event|Proposition)" in prompt.text
    assert "confidence 0.6..1" in prompt.text
    assert "AliasOf(" not in prompt.text
    assert "<example_source_text>\nNora sang.\n" in prompt.text
    assert "<example_source_text>\nAlthough it snowed, Nora sang.\n" in prompt.text
    assert "<example_source_text>\nIf it rains, Nora will leave.\n" in prompt.text


def test_prompt_preserves_source_text_and_semantic_rules() -> None:
    source = "  The door was opened by Alice.\nCafé ☕ remained open.  "

    prompt = build_catalog_document_prompt(source, catalog=load_ontology_catalog())

    assert (
        prompt.text.split("<source_text>\n", 1)[1].split("\n</source_text>", 1)[0]
        == source
    )
    assert "Active and\n  passive descriptions" in prompt.text
    assert "Do not invent an agent for an agentless passive" in prompt.text
    assert "ProvisionalRelation" in prompt.text
    assert "legacy Evaluation fallback" in prompt.text
    assert "Do not output span numbers" in prompt.text
    assert "extracted-hypothesis" in prompt.text
    assert "Classify each assertion separately" in prompt.text
    assert "conditional_on" in prompt.text


def test_prompt_uses_only_selected_ontology_modules() -> None:
    catalog = load_ontology_catalog(modules=("upper", "discourse"))

    prompt = build_catalog_document_prompt(
        "Mira left despite the rain.", catalog=catalog
    )

    assert "Concession(statement: Clause, counter_statement: Clause)" in prompt.text
    assert "Kick(" not in prompt.text
    assert "lexical-v1" not in prompt.text


def test_prompt_excludes_examples_requiring_unselected_modules() -> None:
    prompt = build_catalog_document_prompt(
        "Mira left.", catalog=load_ontology_catalog(modules=("upper",))
    )

    assert "<example_source_text>\nNora sang.\n" in prompt.text
    assert "<example_source_text>\nAlthough it snowed, Nora sang.\n" not in prompt.text
    assert "Concession(" not in prompt.text


def test_prompt_rejects_empty_source_text() -> None:
    with pytest.raises(ValueError, match="source_text"):
        build_catalog_document_prompt(" \t ", catalog=load_ontology_catalog())


def test_existing_relation_guidance_does_not_register_catalog_predicates() -> None:
    catalog = load_ontology_catalog(modules=("upper",))
    prompt = build_catalog_document_prompt("The box is near the door.", catalog=catalog)
    assert "Near(entity, nearby_entity)" in prompt.text
    assert "These descriptions do not add predicates" in prompt.text
    assert "Near" not in catalog.relations


def test_causal_guidance_reuses_legacy_meaning_with_catalog_signature() -> None:
    catalog = load_ontology_catalog()
    guidance = catalog_prompt._relation_guidance(catalog)

    assert (
        "Causes(cause, effect) [catalog predicate; guidance reused from Cause]"
        in guidance
    )
    assert "Do not use merely because X performs an action that affects Y." in guidance
    assert "- Cause(cause, effect):" not in guidance
    assert "Cause" not in catalog.relations
    assert catalog.signatures()["Causes"] == (
        frozenset({"Event", "Proposition"}),
        frozenset({"Event", "Proposition"}),
    )


def test_reviewed_equivalents_have_worked_examples_without_legacy_duplicates() -> None:
    examples = [
        json.loads(line)
        for line in catalog_prompt._PROMPT_EXAMPLES_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for legacy, target in catalog_prompt._LEGACY_CATALOG_EQUIVALENTS.items():
        matching = [
            example
            for example in examples
            if any(a["predicate"] == target for a in example["assertions"])
        ]
        assert matching, f"Missing worked example for {legacy} -> {target}"
        for example in matching:
            assert not any(
                a["predicate"] == "ProvisionalRelation"
                and a["provisional_relation_id"].casefold() == legacy.casefold()
                for a in example["assertions"]
            )


def test_every_legacy_relation_description_is_reused_but_not_operators() -> None:
    catalog = load_ontology_catalog()
    guidance = catalog_prompt._relation_guidance(catalog)
    raw = yaml.safe_load(
        catalog_prompt._PREDICATE_CONFIG_PATH.read_text(encoding="utf-8")
    )

    for name, definition in raw["predicates"].items():
        assert " ".join(definition["description"].split()) in guidance
        if name != "Cause" and not definition.get("fallback"):
            assert f"- {name}({', '.join(definition['roles'])}):" in guidance
    assert "- ArgMin(" not in guidance
    assert "Legacy Evaluation fallback semantics:" in guidance
    assert "variable_arity" not in guidance


def test_inactive_equivalent_is_only_provisional_guidance() -> None:
    catalog = load_ontology_catalog()
    relations = {
        name: value for name, value in catalog.relations.items() if name != "Causes"
    }
    guidance = catalog_prompt._relation_guidance(replace(catalog, relations=relations))
    assert "- Cause(cause, effect):" in guidance
    assert "guidance reused from Cause" not in guidance


def test_reviewed_equivalence_rejects_changed_catalog_argument_order() -> None:
    catalog = load_ontology_catalog()
    relation = catalog.relations["Causes"]
    relations = dict(catalog.relations)
    relations["Causes"] = replace(relation, arguments=relation.arguments[::-1])

    with pytest.raises(ValueError, match="arity or ordered argument roles"):
        catalog_prompt._relation_guidance(replace(catalog, relations=relations))


def test_same_name_is_not_an_implicit_semantic_mapping() -> None:
    catalog = load_ontology_catalog()
    relations = dict(catalog.relations)
    relations["Near"] = replace(catalog.relations["Causes"], name="Near")

    with pytest.raises(ValueError, match="explicit reviewed catalog equivalence"):
        catalog_prompt._relation_guidance(replace(catalog, relations=relations))


@pytest.mark.parametrize("mutation", ["arity", "roles", "duplicate_roles"])
def test_invalid_legacy_role_signature_is_rejected(tmp_path, monkeypatch, mutation):
    raw = yaml.safe_load(
        catalog_prompt._PREDICATE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    if mutation == "arity":
        raw["predicates"]["Near"]["arity"] = 3
    elif mutation == "roles":
        raw["predicates"]["Near"]["roles"] = ["entity", ""]
    else:
        raw["predicates"]["Near"]["roles"] = ["entity", "entity"]
    schema = tmp_path / "predicates.yaml"
    schema.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(catalog_prompt, "_PREDICATE_CONFIG_PATH", schema)

    with pytest.raises(ValueError, match="roles"):
        catalog_prompt._relation_guidance(load_ontology_catalog())


def test_legacy_description_change_updates_prompt_identity(tmp_path, monkeypatch):
    catalog = load_ontology_catalog()
    first = build_catalog_document_prompt("Nora left.", catalog=catalog)
    raw = yaml.safe_load(
        catalog_prompt._PREDICATE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    raw["predicates"]["Cause"]["description"] += " Explicit revised causal guidance."
    schema = tmp_path / "predicates.yaml"
    schema.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(catalog_prompt, "_PREDICATE_CONFIG_PATH", schema)

    second = build_catalog_document_prompt("Nora left.", catalog=catalog)
    assert first.version != second.version
    assert "Explicit revised causal guidance." in second.text


def test_every_rendered_example_matches_the_decoding_schema() -> None:
    catalog = load_ontology_catalog()
    prompt = build_catalog_document_prompt("The door opened.", catalog=catalog)
    validator = jsonschema.Draft202012Validator(catalog_output_schema(catalog))
    examples = re.findall(
        r"<example_output_json>\n(.*?)\n</example_output_json>", prompt.text, re.S
    )
    sources = re.findall(
        r"<example_source_text>\n(.*?)\n</example_source_text>", prompt.text, re.S
    )
    assert examples
    for index, (source, example) in enumerate(zip(sources, examples, strict=True)):
        payload = json.loads(example)
        validator.validate(payload)
        prepare_catalog_target(
            payload,
            document_id=f"example-{index}",
            corpus_id="prompt-examples",
            source_text=source,
            parser_provenance={
                "model": "fixture",
                "parser_version": "fixture-v1",
                "prompt_version": prompt.version,
                "ontology_versions": catalog.versions,
                "provider": "fixture",
            },
            catalog=catalog,
        )


def test_prompt_identity_excludes_source_text() -> None:
    catalog = load_ontology_catalog()
    first = build_catalog_document_prompt("Nora left.", catalog=catalog)
    second = build_catalog_document_prompt("Ivo arrived.", catalog=catalog)
    assert first.version == second.version
    assert first.text != second.text


def test_parser_exposes_the_active_prompt_identity_without_generation() -> None:
    catalog = load_ontology_catalog(modules=("upper",))
    parser = ReferenceSemanticParser(
        backend=CallableBackend(lambda **_: pytest.fail("Identity must not generate")),
        config=SemanticParserConfig(
            model_name="identity-test", prompt_version="historical-label"
        ),
        catalog=catalog,
    )
    expected = build_catalog_document_prompt("Any source.", catalog=catalog).version
    assert parser.prompt_version == expected
    assert parser.prompt_version != "historical-label"


@pytest.mark.parametrize(
    "component",
    [
        "_catalog_guide",
        "load_shared_extraction_guidance",
        "_relation_guidance",
        "_worked_examples",
    ],
)
def test_prompt_identity_tracks_every_guidance_input(monkeypatch, component) -> None:
    catalog = load_ontology_catalog()
    first = build_catalog_document_prompt("Nora left.", catalog=catalog)
    original = getattr(catalog_prompt, component)
    monkeypatch.setattr(
        catalog_prompt,
        component,
        lambda *a, **kw: original(*a, **kw) + "\nChanged guidance.",
    )
    second = build_catalog_document_prompt("Nora left.", catalog=catalog)
    assert second.version != first.version
