"""Deterministic normalization for structured semantic output."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from parser.semantic.schema import SemanticAssertion, SemanticParseResult

_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_PREFIX_RE = re.compile(r"^[0-9]")

_TEMPORAL_ROLES = {
    "time",
    "start_time",
    "end_time",
    "time_or_event",
}


class SemanticNormalizationError(ValueError):
    """Raised when semantic values cannot be normalized safely."""


def normalize_document_target(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize representation tokens without rewriting evidence or identity.

    Catalog names and enum values allow surrounding whitespace cleanup only.
    Entity IDs, references, quotations, alternatives, and provisional relation
    identifiers remain exact. In particular, this does not merge aliases,
    reorder arguments, infer factuality, or repair an incorrect entity type.
    """
    normalized = deepcopy(payload)
    entities = normalized.get("entities")
    assertions = normalized.get("assertions")
    if not isinstance(entities, list) or not isinstance(assertions, list):
        raise SemanticNormalizationError(
            "Document entities and assertions must be lists"
        )
    for entity in entities:
        if isinstance(entity, dict) and isinstance(entity.get("type"), str):
            entity["type"] = entity["type"].strip()
    all_assertions = [*assertions]
    rules = normalized.get("rules", [])
    if not isinstance(rules, list):
        raise SemanticNormalizationError("Document rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        variables = rule.get("variables", [])
        if isinstance(variables, list):
            for variable in variables:
                if isinstance(variable, dict) and isinstance(variable.get("type"), str):
                    variable["type"] = variable["type"].strip()
        for side in ("antecedents", "consequents"):
            if isinstance(rule.get(side), list):
                all_assertions.extend(rule[side])
    for assertion in all_assertions:
        if not isinstance(assertion, dict):
            continue
        for field in ("predicate", "status", "factuality", "polarity"):
            if isinstance(assertion.get(field), str):
                assertion[field] = assertion[field].strip()
        types = assertion.get("argument_types")
        if isinstance(types, list):
            assertion["argument_types"] = [
                value.strip() if isinstance(value, str) else value for value in types
            ]
        # Absence denotes no explicit condition reference, never a guessed one.
        assertion.setdefault("conditional_on", None)
    return normalized


def _normalize_phrase(value: str, field: str) -> str:
    """Normalize Unicode and whitespace without changing letter case."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    if not normalized:
        raise SemanticNormalizationError(f"{field} cannot be empty")

    return normalized


def normalize_symbol(value: str, field: str = "symbol") -> str:
    """Convert a phrase into one MeTTa-safe symbol.

    This performs deterministic representation cleanup:
    - Unicode is normalized
    - whitespace becomes underscores
    - commas in numeric values are removed
    - decimal points become underscores
    - percent signs become ``percent``
    - apostrophes and other punctuation become separators
    - unsupported Unicode characters are transliterated where possible
    - repeated underscores are collapsed
    """
    normalized = _normalize_phrase(value, field)
    if (
        normalized.startswith("$")
        and len(normalized) > 1
        and (normalized[1].isalpha() or normalized[1] == "_")
    ):
        variable_name = normalized[1:]

        variable_name = unicodedata.normalize("NFKD", variable_name)
        variable_name = variable_name.encode("ascii", "ignore").decode("ascii")

        variable_name = re.sub(
            r"[^A-Za-z0-9_]+",
            "_",
            variable_name,
        )

        variable_name = re.sub(
            r"_+",
            "_",
            variable_name,
        ).strip("_")

        if not variable_name:
            raise SemanticNormalizationError(f"{field} variable cannot be empty")

        return f"${variable_name}"
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("‘", "'")

    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")

    normalized = normalized.replace(",", "")
    normalized = normalized.replace("%", "_percent")
    normalized = normalized.replace(".", "_")
    normalized = normalized.replace("&", "_and_")

    normalized = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        normalized,
    )

    normalized = re.sub(r"_+", "_", normalized).strip("_")

    if not normalized:
        raise SemanticNormalizationError(f"{field} cannot be empty")

    return normalized


def normalize_argument_value(value: str, role: str) -> str:
    """Normalize an argument into a safe MeTTa symbol.

    Temporal numeric values receive a ``time_`` prefix.
    Other values beginning with a digit receive a ``num_`` prefix.

    Examples:
        2025, role=time       -> time_2025
        10 AM, role=time      -> time_10_AM
        25 liters             -> num_25_liters
        1,000 employees       -> num_1000_employees
        37.5 degrees          -> num_37_5_degrees
        25%                   -> num_25_percent
    """
    normalized = normalize_symbol(value, "argument value")
    role_normalized = role.casefold()

    if normalized.startswith("time_") or normalized.startswith("num_"):
        return normalized

    if _NUMERIC_PREFIX_RE.match(normalized):
        if role_normalized in _TEMPORAL_ROLES:
            return f"time_{normalized}"

        return f"num_{normalized}"

    return normalized


def _normalize_assertion(
    assertion: SemanticAssertion,
    *,
    normalized_aliases: Mapping[str, str],
    normalized_types: Mapping[str, str],
) -> None:
    """Normalize one semantic assertion in place."""
    if assertion.relation is not None:
        assertion.relation = normalize_symbol(
            assertion.relation.casefold(),
            "relation",
        )

    for argument in assertion.arguments:
        entity = _normalize_phrase(
            argument.value,
            "argument value",
        )

        entity = normalized_aliases.get(entity, entity)

        expected_type = normalized_types.get(entity)

        if (
            expected_type is not None
            and argument.type is not None
            and _normalize_phrase(
                argument.type,
                "argument type",
            ).casefold()
            != expected_type.casefold()
        ):
            raise SemanticNormalizationError(
                f"Alias/type conflict for {entity!r}: expected "
                f"{expected_type!r}, got {argument.type!r}"
            )

        argument.role = _normalize_phrase(
            argument.role,
            "argument role",
        ).casefold()

        argument.value = normalize_argument_value(
            entity,
            argument.role,
        )

        if argument.type is not None:
            argument.type = _normalize_phrase(
                argument.type,
                "argument type",
            )

    assertion.source_span = _normalize_phrase(
        assertion.source_span,
        "source span",
    )

    assertion.alternatives = [
        _normalize_phrase(
            alternative,
            "alternative",
        )
        for alternative in assertion.alternatives
    ]


def normalize_semantic_result(
    result: SemanticParseResult,
    *,
    aliases: Mapping[str, str] | None = None,
    alias_types: Mapping[str, str] | None = None,
) -> SemanticParseResult:
    """Return a normalized deep copy of structured semantic output.

    Aliases are explicit rather than guessed. Keys are observed entity names
    and values are their canonical names. Entity case is preserved.
    """
    normalized_result = result.model_copy(deep=True)

    normalized_aliases = {
        _normalize_phrase(alias, "alias"): _normalize_phrase(
            canonical,
            "alias target",
        )
        for alias, canonical in (aliases or {}).items()
    }

    normalized_types = {
        _normalize_phrase(entity, "typed entity"): _normalize_phrase(
            entity_type,
            "entity type",
        )
        for entity, entity_type in (alias_types or {}).items()
    }

    for assertion in normalized_result.assertions:
        _normalize_assertion(
            assertion,
            normalized_aliases=normalized_aliases,
            normalized_types=normalized_types,
        )

    for rule in normalized_result.rules:
        rule.source_span = _normalize_phrase(
            rule.source_span,
            "rule source span",
        )

        for assertion in rule.antecedents:
            _normalize_assertion(
                assertion,
                normalized_aliases=normalized_aliases,
                normalized_types=normalized_types,
            )

        for assertion in rule.consequents:
            _normalize_assertion(
                assertion,
                normalized_aliases=normalized_aliases,
                normalized_types=normalized_types,
            )

    return normalized_result
