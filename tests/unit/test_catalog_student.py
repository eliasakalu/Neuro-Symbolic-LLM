"""Catalog teacher labels and student prompts share one validated target."""

import json
from pathlib import Path

import pytest

pytest.importorskip("hybrid_miner.contracts.semantic_document")

from parser.ontology.catalog import load_ontology_catalog  # noqa: E402
from parser.semantic.catalog_prompt import build_catalog_document_prompt  # noqa: E402
from parser.semantic.catalog_student import (  # noqa: E402
    validate_catalog_student_checkpoint,
    validate_catalog_student_pair,
)
from parser.semantic.dataset import StudentDataset  # noqa: E402


def _pair() -> dict[str, object]:
    return {
        "contract_version": "semantic-document-v1",
        "document_id": "doc-1",
        "corpus_id": "corpus-1",
        "input": "Mira left.",
        "parser": {
            "model": "teacher-model",
            "parser_version": "catalog-semantic-parser-v4",
            "prompt_version": build_catalog_document_prompt(
                "Mira left.", catalog=load_ontology_catalog(modules=("upper",))
            ).version,
            "ontology_versions": ["upper-v1"],
            "provider": "test",
            "model_revision": None,
        },
        "label": json.dumps(
            {
                "entities": [
                    {
                        "id": "leaving",
                        "type": "Event",
                        "mention": "left",
                        "occurrence": 0,
                    }
                ],
                "assertions": [
                    {
                        "predicate": "Event",
                        "arguments": ["leaving"],
                        "argument_types": ["Event"],
                        "status": "extracted-hypothesis",
                        "factuality": "asserted",
                        "confidence": 0.9,
                        "source_quotes": [{"text": "left", "occurrence": 0}],
                        "alternatives": [],
                    }
                ],
            }
        ),
    }


def test_pair_rebuilds_the_miner_document_from_the_active_prompt() -> None:
    catalog = load_ontology_catalog(modules=("upper",))

    prompt, label, document = validate_catalog_student_pair(_pair(), catalog=catalog)

    assert "Ontology versions: upper-v1" in prompt
    assert json.loads(label)["assertions"][0]["predicate"] == "Event"
    assert document.assertions[0].source_spans[0].start == 5
    assert document.source_text == "Mira left."


def test_pair_rejects_stale_prompt_provenance() -> None:
    pair = _pair()
    provenance = pair["parser"]
    assert isinstance(provenance, dict)
    provenance["prompt_version"] = "catalog-document-v2"

    with pytest.raises(ValueError, match="prompt version"):
        validate_catalog_student_pair(
            pair, catalog=load_ontology_catalog(modules=("upper",))
        )


def test_default_fields_do_not_change_exact_target_comparison():
    from parser.semantic.catalog_student import document_target

    catalog = load_ontology_catalog(modules=("upper",))
    _, _, document = validate_catalog_student_pair(_pair(), catalog=catalog)
    recovered = {**_pair(), "label": json.dumps(document_target(document))}
    _, _, restored = validate_catalog_student_pair(recovered, catalog=catalog)
    assert document_target(restored) == document_target(document)


def test_rendering_does_not_silently_drop_factuality():
    from parser.semantic.compatibility import render_document_view

    pair = _pair()
    payload = json.loads(pair["label"])
    payload["assertions"][0]["factuality"] = "speculative"
    pair["label"] = json.dumps(payload)
    _, _, document = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog(modules=("upper",))
    )
    before = document.canonical_json()
    with pytest.raises(ValueError, match="factuality"):
        render_document_view(document)
    assert render_document_view(document, asserted_only=True) == []
    assert document.canonical_json() == before


def test_compatible_label_retains_its_original_parser_provenance() -> None:
    pair = _pair()
    provenance = pair["parser"]
    assert isinstance(provenance, dict)
    provenance["parser_version"] = "catalog-semantic-parser-v3"
    _, _, document = validate_catalog_student_pair(
        pair, catalog=load_ontology_catalog(modules=("upper",))
    )
    assert document.parser.parser_version == "catalog-semantic-parser-v3"


def test_pair_rejects_label_without_factuality() -> None:
    pair = _pair()
    raw_label = pair["label"]
    assert isinstance(raw_label, str)
    label = json.loads(raw_label)
    del label["assertions"][0]["factuality"]
    pair["label"] = json.dumps(label)

    with pytest.raises(ValueError, match="explicit factuality"):
        validate_catalog_student_pair(
            pair, catalog=load_ontology_catalog(modules=("upper",))
        )


def test_training_masks_prompt_and_preserves_the_validated_target() -> None:
    class WordTokenizer:
        eos_token_id = 1

        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": [index + 2 for index, _ in enumerate(text.split())]}

    pair = _pair()
    catalog = load_ontology_catalog(modules=("upper",))
    prompt, label, _ = validate_catalog_student_pair(pair, catalog=catalog)
    tokenizer = WordTokenizer()
    dataset = StudentDataset([pair], tokenizer, max_length=16384, catalog=catalog)

    assert len(dataset) == 1
    prompt_length = len(tokenizer(prompt)["input_ids"])
    labels = dataset[0]["labels"].tolist()
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == tokenizer(label)["input_ids"] + [1]


def test_student_checkpoint_requires_matching_prompt_and_ontology(
    tmp_path: Path,
) -> None:
    catalog = load_ontology_catalog(modules=("upper",))
    metadata_path = tmp_path / "training_metadata.json"
    metadata = {
        "target_contract_version": "semantic-document-v1",
        "prompt_version": build_catalog_document_prompt(
            "Mira left.", catalog=catalog
        ).version,
        "ontology_versions": ["upper-v1"],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    validate_catalog_student_checkpoint(tmp_path, catalog=catalog)

    metadata["prompt_version"] = "catalog-document-v4"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong prompt version"):
        validate_catalog_student_checkpoint(tmp_path, catalog=catalog)


def test_pair_rejects_changed_guidance_with_unchanged_configured_version(
    monkeypatch,
) -> None:
    from parser.semantic import catalog_prompt

    pair = _pair()
    original = catalog_prompt.load_shared_extraction_guidance
    monkeypatch.setattr(
        catalog_prompt,
        "load_shared_extraction_guidance",
        lambda: original() + "\nChanged roles.",
    )
    with pytest.raises(ValueError, match="prompt version"):
        validate_catalog_student_pair(
            pair, catalog=load_ontology_catalog(modules=("upper",))
        )
