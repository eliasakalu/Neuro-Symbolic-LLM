"""Exact source-quote resolution before semantic-document validation."""

import pytest

from parser.semantic.catalog_source_anchors import resolve_catalog_source_anchors


def test_derives_unicode_offsets_from_exact_source_quotes() -> None:
    source = "\U0001f642 Mira left. Mira returned."
    draft = {
        "entities": [
            {"id": "first", "type": "Entity", "mention": "Mira", "occurrence": 0},
            {"id": "second", "type": "Entity", "mention": "Mira", "occurrence": 1},
            {
                "id": "implicit",
                "type": "Proposition",
                "mention": None,
                "occurrence": None,
            },
        ],
        "assertions": [
            {
                "predicate": "SomePredicate",
                "source_quotes": [{"text": "Mira returned", "occurrence": 0}],
            }
        ],
    }

    anchored = resolve_catalog_source_anchors(draft, source_text=source)

    assert anchored["entities"][0]["span"] == {"start": 2, "end": 6}
    assert anchored["entities"][1]["span"] == {"start": 13, "end": 17}
    assert anchored["entities"][2]["span"] is None
    assert anchored["assertions"][0]["source_spans"] == [{"start": 13, "end": 26}]
    assert "source_quotes" not in anchored["assertions"][0]


@pytest.mark.parametrize(
    ("anchor", "error"),
    [
        ({"text": "Mira", "occurrence": 2}, "is absent from source text"),
        ({"text": "mira", "occurrence": 0}, "is absent from source text"),
        ({"text": "Mira", "occurrence": True}, "nonnegative integer"),
        ({"text": "", "occurrence": 0}, "nonempty source text"),
        ({"text": "Mira", "occurrence": -1}, "nonnegative integer"),
    ],
)
def test_rejects_unresolvable_or_invalid_anchors(
    anchor: dict[str, object], error: str
) -> None:
    draft = {
        "entities": [
            {
                "id": "mira",
                "type": "Entity",
                "mention": anchor["text"],
                "occurrence": anchor["occurrence"],
            }
        ],
        "assertions": [],
    }

    with pytest.raises(ValueError, match=error):
        resolve_catalog_source_anchors(draft, source_text="Mira left.")


def test_rejects_model_supplied_numeric_spans() -> None:
    draft = {
        "entities": [],
        "assertions": [
            {
                "predicate": "Event",
                "source_quotes": [{"text": "left", "occurrence": 0}],
                "source_spans": [{"start": 0, "end": 4}],
            }
        ],
    }

    with pytest.raises(ValueError, match="must not supply source_spans"):
        resolve_catalog_source_anchors(draft, source_text="Mira left.")


def test_rule_and_signed_literals_retain_their_own_source_evidence() -> None:
    source = "All red things are not blue."
    draft = {
        "entities": [],
        "assertions": [],
        "rules": [
            {
                "id": "rule-1",
                "variables": [{"id": "$x", "type": "Entity"}],
                "source_quotes": [{"text": source, "occurrence": 0}],
                "antecedents": [
                    {
                        "predicate": "ProvisionalRelation",
                        "source_quotes": [{"text": "red things", "occurrence": 0}],
                    }
                ],
                "consequents": [
                    {
                        "predicate": "ProvisionalRelation",
                        "polarity": "negative",
                        "source_quotes": [{"text": "not blue", "occurrence": 0}],
                    }
                ],
            }
        ],
    }
    anchored = resolve_catalog_source_anchors(draft, source_text=source)
    rule = anchored["rules"][0]
    assert rule["source_spans"] == [{"start": 0, "end": len(source)}]
    consequent = rule["consequents"][0]
    assert consequent["polarity"] == "negative"
    assert consequent["source_spans"] == [
        {
            "start": source.index("not blue"),
            "end": source.index("not blue") + len("not blue"),
        }
    ]
    assert "source_quotes" not in consequent
