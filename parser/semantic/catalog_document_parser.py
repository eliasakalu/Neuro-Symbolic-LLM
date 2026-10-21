"""Compatibility constructor and validation for the miner-facing document target."""

from __future__ import annotations

from hybrid_miner.contracts.semantic_document import SemanticDocument
from pydantic import ValidationError

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.backends import ModelBackend
from parser.semantic.catalog_source_anchors import resolve_catalog_source_anchors
from parser.semantic.normalization import normalize_document_target
from parser.semantic.semantic_parser import (
    ReferenceSemanticParser,
    SemanticParseError,
    SemanticParserConfig,
)

CATALOG_PARSER_VERSION = "catalog-semantic-parser-v7"


class CatalogSemanticDocumentParser(ReferenceSemanticParser):
    """Preserve the public constructor; extraction runs in the shared parser.

    New integrations may configure ReferenceSemanticParser directly. This class
    adds no generation, normalization, or document-parsing implementation.
    """

    def __init__(
        self,
        *,
        backend: ModelBackend,
        model_name: str,
        catalog: OntologyCatalog | None = None,
        model_revision: str | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            config=SemanticParserConfig(model_name=model_name),
            catalog=catalog,
            model_revision=model_revision,
        )


def validate_catalog_target(
    payload: dict[str, object],
    *,
    document_id: str,
    corpus_id: str,
    source_text: str,
    parser_provenance: dict[str, object],
    catalog: OntologyCatalog,
) -> SemanticDocument:
    """Validate a teacher or student target through the identical S1 contract."""
    document, _ = prepare_catalog_target(
        payload,
        document_id=document_id,
        corpus_id=corpus_id,
        source_text=source_text,
        parser_provenance=parser_provenance,
        catalog=catalog,
    )
    return document


def prepare_catalog_target(
    payload: dict[str, object],
    *,
    document_id: str,
    corpus_id: str,
    source_text: str,
    parser_provenance: dict[str, object],
    catalog: OntologyCatalog,
) -> tuple[SemanticDocument, dict[str, object]]:
    """Return the validated document and the normalized draft used to build it."""

    try:
        normalized = normalize_document_target(payload)
        anchored = resolve_catalog_source_anchors(normalized, source_text=source_text)
        document = SemanticDocument.model_validate(
            {
                "document_id": document_id,
                "corpus_id": corpus_id,
                "source_text": source_text,
                "parser": parser_provenance,
                "entities": anchored["entities"],
                "assertions": anchored["assertions"],
                "rules": anchored.get("rules", []),
            }
        )
        document.validate_against_signatures(
            ontology_versions=catalog.versions,
            signatures=catalog.signatures(),
        )
        _validate_catalog_constraints(document, catalog)
    except (ValidationError, ValueError) as error:
        raise SemanticParseError(f"Invalid semantic document: {error}") from error
    return document, normalized


def _validate_catalog_constraints(
    document: SemanticDocument, catalog: OntologyCatalog
) -> None:
    """Enforce catalog limits not covered by the interchange contract."""

    allowed_types = catalog.types | {"String"}
    for entity in document.entities:
        if entity.type not in allowed_types:
            raise ValueError(f"Undeclared ontology entity type {entity.type!r}")
    assertions = [*document.assertions]
    for rule in document.rules:
        for variable in rule.variables:
            if variable.type not in allowed_types:
                raise ValueError(f"Undeclared rule variable type {variable.type!r}")
        assertions.extend((*rule.antecedents, *rule.consequents))
    for assertion in assertions:
        if assertion.factuality is None:
            raise ValueError("Catalog assertions require explicit factuality")
        if assertion.predicate == "ProvisionalRelation":
            continue
        lower, upper = catalog.relation(assertion.predicate).confidence_range
        if not lower <= assertion.confidence <= upper:
            raise ValueError(
                f"{assertion.predicate} confidence {assertion.confidence} "
                f"is outside catalog range [{lower}, {upper}]"
            )
