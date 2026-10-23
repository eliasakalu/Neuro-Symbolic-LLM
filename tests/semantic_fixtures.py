"""Synthetic contract fixtures, not extraction-quality ground truth."""

import json

from parser.ontology.catalog import load_ontology_catalog
from parser.semantic.catalog_document_parser import prepare_catalog_target
from parser.semantic.catalog_prompt import build_catalog_document_prompt
from parser.semantic.catalog_student import make_student_pair


def target(text):
    return {
        "entities": [{"id": "e1", "type": "Event", "mention": text, "occurrence": 0}],
        "assertions": [
            {
                "predicate": "Event",
                "arguments": ["e1"],
                "argument_types": ["Event"],
                "status": "extracted-hypothesis",
                "factuality": "asserted",
                "confidence": 0.95,
                "source_quotes": [{"text": text, "occurrence": 0}],
                "alternatives": [],
            }
        ],
        "rules": [],
    }


def pair(text="Mira left.", *, catalog=None):
    catalog = catalog if catalog is not None else load_ontology_catalog()
    document, payload = prepare_catalog_target(
        target(text),
        document_id="fixture",
        corpus_id="fixtures",
        source_text=text,
        parser_provenance={
            "model": "fixture",
            "provider": "fixture",
            "parser_version": "fixture",
            "prompt_version": build_catalog_document_prompt(
                text, catalog=catalog
            ).version,
            "ontology_versions": list(catalog.versions),
        },
        catalog=catalog,
    )
    return make_student_pair(document, target=payload)


def write_checkpoint_metadata(path):
    catalog = load_ontology_catalog()
    (path / "training_metadata.json").write_text(
        json.dumps(
            {
                "target_contract_version": "semantic-document-v1",
                "prompt_version": build_catalog_document_prompt(
                    "Checkpoint.", catalog=catalog
                ).version,
                "ontology_versions": list(catalog.versions),
            }
        ),
        encoding="utf-8",
    )
