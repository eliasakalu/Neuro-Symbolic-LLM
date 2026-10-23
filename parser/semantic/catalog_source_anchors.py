"""Resolve model-quoted source evidence into contract source spans."""

from __future__ import annotations

from typing import Any


def _source_span(anchor: object, *, source_text: str, context: str) -> dict[str, int]:
    if not isinstance(anchor, dict) or set(anchor) != {"text", "occurrence"}:
        raise ValueError(f"{context} must contain text and occurrence")
    quote = anchor["text"]
    occurrence = anchor["occurrence"]
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError(f"{context} text must be nonempty source text")
    if (
        not isinstance(occurrence, int)
        or isinstance(occurrence, bool)
        or occurrence < 0
    ):
        raise ValueError(f"{context} occurrence must be a nonnegative integer")

    positions: list[int] = []
    start = 0
    while (position := source_text.find(quote, start)) != -1:
        positions.append(position)
        start = position + 1
    if occurrence >= len(positions):
        raise ValueError(
            f"{context} quote {quote!r} occurrence {occurrence} "
            "is absent from source text"
        )
    position = positions[occurrence]
    return {"start": position, "end": position + len(quote)}


def resolve_catalog_source_anchors(
    payload: dict[str, object], *, source_text: str
) -> dict[str, list[dict[str, Any]]]:
    """Convert an untrusted quote-based draft to miner-contract fields.

    Quotes are exact Unicode substrings; occurrence is zero-based among exact
    matches. No source text is normalized and no model-provided offsets survive.
    """

    raw_entities = payload["entities"]
    raw_assertions = payload["assertions"]
    if not isinstance(raw_entities, list) or not isinstance(raw_assertions, list):
        raise ValueError("Catalog entities and assertions must be lists")

    entities: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_entities):
        context = f"entity {index}"
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "type",
            "mention",
            "occurrence",
        }:
            raise ValueError(
                f"{context} must contain id, type, mention, and occurrence"
            )
        mention = raw["mention"]
        occurrence = raw["occurrence"]
        if mention is None and occurrence is None:
            span = None
        elif mention is not None:
            span = _source_span(
                {"text": mention, "occurrence": occurrence},
                source_text=source_text,
                context=context,
            )
        else:
            raise ValueError(f"{context} occurrence requires a mention")
        entities.append({"id": raw["id"], "type": raw["type"], "span": span})

    assertions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_assertions):
        context = f"assertion {index}"
        if not isinstance(raw, dict) or "source_quotes" not in raw:
            raise ValueError(f"{context} must contain source_quotes")
        quotes = raw["source_quotes"]
        if not isinstance(quotes, list) or not quotes:
            raise ValueError(f"{context} source_quotes must be a nonempty list")
        if "source_spans" in raw:
            raise ValueError(f"{context} must not supply source_spans")
        assertions.append(
            {
                **{key: value for key, value in raw.items() if key != "source_quotes"},
                "source_spans": [
                    _source_span(
                        quote,
                        source_text=source_text,
                        context=f"{context} source quote {quote_index}",
                    )
                    for quote_index, quote in enumerate(quotes)
                ],
            }
        )
    result = {"entities": entities, "assertions": assertions}
    if "rules" in payload:
        raw_rules = payload["rules"]
        if not isinstance(raw_rules, list):
            raise ValueError("Document rules must be a list")
        rules = []
        for index, raw in enumerate(raw_rules):
            if not isinstance(raw, dict):
                raise ValueError(f"rule {index} must be an object")
            quotes = raw.get("source_quotes")
            if not isinstance(quotes, list) or not quotes or "source_spans" in raw:
                raise ValueError(
                    f"rule {index} requires source_quotes, not source_spans"
                )
            rule = {key: value for key, value in raw.items() if key != "source_quotes"}
            rule["source_spans"] = [
                _source_span(quote, source_text=source_text, context=f"rule {index}")
                for quote in quotes
            ]
            for side in ("antecedents", "consequents"):
                clauses = raw.get(side)
                if not isinstance(clauses, list) or not clauses:
                    raise ValueError(f"rule {index} requires nonempty {side}")
                rule[side] = resolve_catalog_source_anchors(
                    {"entities": [], "assertions": clauses}, source_text=source_text
                )["assertions"]
            rules.append(rule)
        result["rules"] = rules
    return result
