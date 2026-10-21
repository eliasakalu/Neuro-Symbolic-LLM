"""Derive the quote-based decoding schema from the shared miner contract."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hybrid_miner.contracts.semantic_document import SemanticDocument

from parser.ontology.catalog import OntologyCatalog


def _scope_variants(assertion: dict[str, Any]) -> list[dict[str, Any]]:
    """Encode the contract's conditional_on/factuality relationship."""
    unconditional = deepcopy(assertion)
    unconditional["properties"]["factuality"] = {
        "type": "string",
        "enum": ["asserted", "opinion", "speculative"],
    }
    unconditional["properties"]["conditional_on"] = {"type": "null"}
    hypothetical = deepcopy(assertion)
    hypothetical["properties"]["factuality"] = {"const": "hypothetical"}
    hypothetical["properties"]["conditional_on"] = {
        "anyOf": [
            {"type": "string", "minLength": 1, "pattern": r"^[^\s]+$"},
            {"type": "null"},
        ]
    }
    return [unconditional, hypothetical]


def catalog_output_schema(catalog: OntologyCatalog) -> dict[str, Any]:
    """Constrain model output without duplicating the final document schema.

    The model copies quotations; deterministic code constructs source spans.
    Predicate branches also constrain arity, ordered argument types, confidence
    bounds, and provisional metadata using the active catalog. Reference and
    source consistency still require validation after generation; grammar
    constraints do not establish semantic correctness.
    """
    document_schema = deepcopy(SemanticDocument.model_json_schema())
    definitions = document_schema["$defs"]
    entity = definitions["EntityRecord"]
    entity["properties"].pop("span")
    entity["properties"].update(
        {
            "type": {"type": "string", "enum": sorted(catalog.types | {"String"})},
            "mention": {
                "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]
            },
            "occurrence": {
                "anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]
            },
        }
    )
    entity["required"] = ["id", "type", "mention", "occurrence"]
    assertion = definitions["SemanticAssertion"]
    assertion["properties"].pop("source_spans")
    assertion["properties"]["source_quotes"] = {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "occurrence": {"type": "integer", "minimum": 0},
            },
            "required": ["text", "occurrence"],
        },
    }
    factuality = assertion["properties"]["factuality"]
    assertion["properties"]["factuality"] = next(
        option for option in factuality["anyOf"] if option.get("type") == "string"
    )
    assertion["required"] = [
        field for field in assertion["required"] if field != "source_spans"
    ] + ["source_quotes", "factuality", "conditional_on"]
    branches = []
    for name, relation in sorted(catalog.relations.items()):
        branch = deepcopy(assertion)
        properties = branch["properties"]
        properties["predicate"] = {"const": name}
        properties["arguments"].update(minItems=relation.arity, maxItems=relation.arity)
        properties["argument_types"] = {
            "type": "array",
            "prefixItems": [
                {"type": "string", "enum": list(argument.allowed_types)}
                for argument in relation.arguments
            ],
            "minItems": relation.arity,
            "maxItems": relation.arity,
        }
        lower, upper = relation.confidence_range
        properties["confidence"].update(minimum=lower, maximum=upper)
        properties["provisional_relation_id"] = {"type": "null"}
        branches.extend(_scope_variants(branch))

    provisional = deepcopy(assertion)
    provisional["properties"]["predicate"] = {"const": "ProvisionalRelation"}
    provisional["properties"]["provisional_relation_id"] = {
        "type": "string",
        "minLength": 1,
        "pattern": r"^[^\s]+$",
    }
    provisional["properties"]["argument_types"]["items"] = {
        "type": "string",
        "enum": sorted(catalog.types | {"String"}),
    }
    provisional["required"].append("provisional_relation_id")
    branches.extend(_scope_variants(provisional))
    definitions["SemanticAssertion"] = {"anyOf": branches}
    # Rule clauses use the same catalog signatures, but cannot assert that
    # their contents occurred. Signed clauses retain explicit source negation.
    rule_branches = []
    for branch in branches:
        factuality_schema = branch["properties"]["factuality"]
        if factuality_schema.get("const") != "hypothetical":
            continue
        literal = deepcopy(branch)
        literal["properties"]["conditional_on"] = {"type": "null"}
        literal["properties"]["polarity"] = {
            "type": "string",
            "enum": ["positive", "negative"],
        }
        literal["required"].append("polarity")
        rule_branches.append(literal)
    definitions["RuleLiteral"] = {"anyOf": rule_branches}
    definitions["RuleVariable"]["properties"]["type"] = {
        "type": "string",
        "enum": sorted(catalog.types | {"String"}),
    }
    rule = definitions["SemanticRule"]
    rule["properties"].pop("source_spans")
    rule["properties"]["source_quotes"] = deepcopy(
        assertion["properties"]["source_quotes"]
    )
    rule["required"] = [
        "id",
        "quantifier",
        "status",
        "variables",
        "antecedents",
        "consequents",
        "source_quotes",
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entities": {"type": "array", "items": {"$ref": "#/$defs/EntityRecord"}},
            "assertions": {
                "type": "array",
                "items": {"$ref": "#/$defs/SemanticAssertion"},
            },
            "rules": {"type": "array", "items": {"$ref": "#/$defs/SemanticRule"}},
        },
        "required": ["entities", "assertions", "rules"],
        "$defs": definitions,
    }
