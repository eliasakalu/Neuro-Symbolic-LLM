"""Adapters for historical assertion records and derived MeTTa rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from parser.semantic.normalization import normalize_semantic_result
from parser.semantic.schema import (
    SemanticArgument,
    SemanticAssertion,
    SemanticParseResult,
    SemanticRule,
)

if TYPE_CHECKING:
    from hybrid_miner.contracts import SemanticDocument


def document_to_legacy_result(
    document: SemanticDocument,
    *,
    aliases: Mapping[str, str] | None = None,
    alias_types: Mapping[str, str] | None = None,
) -> SemanticParseResult:
    """Project a validated document for existing renderers and API consumers.

    Evaluation is a display container, not an ontology promotion. Lexical values
    and symbol normalization make this view unsuitable for round-trip recovery,
    mining, or student labels. The canonical document remains unchanged.
    """
    entities = {entity.id: entity for entity in document.entities}

    def source(spans):
        return document.source_text[
            min(s.start for s in spans) : max(s.end for s in spans)
        ]

    def assertion_view(assertion, variables=frozenset()):
        arguments = []
        for index, (identifier, entity_type) in enumerate(
            zip(assertion.arguments, assertion.argument_types, strict=True)
        ):
            if identifier in variables:
                value = identifier
            else:
                entity = entities[identifier]
                value = (
                    document.source_text[entity.span.start : entity.span.end]
                    if entity.span is not None
                    else identifier
                )
            arguments.append(
                SemanticArgument(
                    value=value, role=f"argument_{index}", type=entity_type
                )
            )
        if assertion.factuality is None:
            raise ValueError("A compatibility view requires explicit source factuality")
        return SemanticAssertion(
            predicate="Evaluation",
            relation=(
                assertion.provisional_relation_id
                if assertion.predicate == "ProvisionalRelation"
                else assertion.predicate
            ),
            arguments=arguments,
            fallback=True,
            polarity=getattr(assertion, "polarity", "positive"),
            factuality=assertion.factuality,
            confidence=assertion.confidence,
            source_span=source(assertion.source_spans),
            alternatives=list(assertion.alternatives),
        )

    result = SemanticParseResult(
        assertions=[assertion_view(a) for a in document.assertions],
        rules=[
            SemanticRule(
                antecedents=[
                    assertion_view(a, frozenset(v.id for v in rule.variables))
                    for a in rule.antecedents
                ],
                consequents=[
                    assertion_view(a, frozenset(v.id for v in rule.variables))
                    for a in rule.consequents
                ],
                source_span=source(rule.source_spans),
            )
            for rule in document.rules
        ],
    )
    normalized = normalize_semantic_result(
        result, aliases=aliases, alias_types=alias_types
    )
    original = [*document.assertions]
    projected = [*normalized.assertions]
    for rule, view_rule in zip(document.rules, normalized.rules, strict=True):
        original.extend((*rule.antecedents, *rule.consequents))
        projected.extend((*view_rule.antecedents, *view_rule.consequents))
    symbols: dict[str, str] = {}
    for assertion, view in zip(original, projected, strict=True):
        for identifier, argument in zip(
            assertion.arguments, view.arguments, strict=True
        ):
            if identifier not in entities:
                continue
            if argument.value in symbols and symbols[argument.value] != identifier:
                raise ValueError(
                    "Distinct entity IDs would collapse in the lexical rendering; use document JSON"
                )
            symbols[argument.value] = identifier
    return normalized


def render_document_view(
    document: SemanticDocument, *, asserted_only: bool = False
) -> list[str]:
    """Render an explicit derived view without modifying canonical documents."""
    from parser.semantic.semantic_parser import ReferenceSemanticParser

    if asserted_only:
        document = document.model_copy(
            update={
                "assertions": tuple(
                    a for a in document.assertions if a.factuality == "asserted"
                ),
                "rules": (),
            }
        )
    if not document.assertions and not document.rules:
        return []
    if any(
        a.conditional_on is not None
        or a.factuality != "asserted"
        or a.predicate == "Negated"
        for a in document.assertions
    ):
        raise ValueError(
            "Lexical MeTTa rendering cannot preserve this assertion's factuality or negation scope; use document JSON"
        )
    view = document_to_legacy_result(document)
    expressions = ReferenceSemanticParser.render_metta(view)
    ReferenceSemanticParser.validate_rendered_metta(expressions)
    return expressions
