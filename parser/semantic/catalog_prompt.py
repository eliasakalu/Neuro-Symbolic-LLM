"""Compose the shared semantic-document prompt from reviewed extraction guidance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from parser.ontology.catalog import OntologyCatalog
from parser.semantic.normalization import normalize_document_target

_PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "parser_config"
    / "catalog_document_prompt.yaml"
)
_PROMPT_EXAMPLES_PATH = _PROMPT_CONFIG_PATH.with_name("catalog_document_examples.jsonl")
_EXTRACTION_CONFIG_PATH = _PROMPT_CONFIG_PATH.with_name("parser_prompt.yaml")
_PREDICATE_CONFIG_PATH = _PROMPT_CONFIG_PATH.with_name("predicate_schema.yaml")

# Reviewed semantic equivalences, not a spelling-based alias rule. The source
# supplies interpretation guidance; the target catalog still owns its types,
# confidence bounds, and admissibility. Argument order must match exactly.
_LEGACY_CATALOG_EQUIVALENTS = {"Cause": "Causes"}


def load_shared_extraction_guidance() -> str:
    """Read target-neutral interpretation rules shared with the existing parser."""
    raw = yaml.safe_load(_EXTRACTION_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("parser"), dict):
        raise ValueError("Parser prompt configuration requires a parser mapping")
    guidance = raw["parser"].get("shared_extraction_guidance")
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("Parser prompt requires shared_extraction_guidance")
    return guidance


@dataclass(frozen=True, slots=True)
class CatalogDocumentPrompt:
    """Rendered prompt and the version to record in parser provenance."""

    version: str
    text: str


def _catalog_guide(catalog: OntologyCatalog) -> str:
    lines = []
    for name, relation in sorted(catalog.relations.items()):
        arguments = ", ".join(
            f"{argument.name}: {'|'.join(argument.allowed_types)}"
            for argument in relation.arguments
        )
        lower, upper = relation.confidence_range
        lines.append(
            f"- {name}({arguments}) [confidence {lower:g}..{upper:g}]: "
            f"{relation.extraction_instruction}"
        )
    return "\n".join(lines)


def _relation_guidance(catalog: OntologyCatalog) -> str:
    """Build interpretation guidance from predicate definitions.

    Only explicitly reviewed equivalents supplement known catalog predicates.
    Other descriptions guide ProvisionalRelation extraction. Legacy operators
    and representation-specific arities are not catalog relation definitions.
    """
    raw = yaml.safe_load(_PREDICATE_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("predicates"), dict):
        raise ValueError("Predicate guidance requires a predicates mapping")
    mapped = []
    provisional = []
    fallback = []
    for name, definition in raw["predicates"].items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(definition, dict)
        ):
            raise ValueError(f"Invalid predicate guidance for {name!r}")
        description = definition.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Predicate guidance requires a description: {name}")
        description = " ".join(description.split())
        if definition.get("fallback"):
            # Reuse the fallback's meaning, not Evaluation's legacy JSON shape
            # or its renderer-specific argument limits.
            fallback.append(f"- {name}: {description}")
            continue
        roles = definition.get("roles")
        arity = definition.get("arity")
        if (
            not isinstance(roles, list)
            or isinstance(arity, bool)
            or not isinstance(arity, int)
            or arity < 1
            or len(roles) != arity
        ):
            raise ValueError(
                f"Predicate guidance requires arity matching ordered roles: {name}"
            )
        if any(not isinstance(role, str) or not role.strip() for role in roles):
            raise ValueError(f"Invalid argument roles for {name!r}")
        if len(set(roles)) != len(roles):
            raise ValueError(f"Duplicate argument roles for {name!r}")
        target = _LEGACY_CATALOG_EQUIVALENTS.get(name)
        if name in catalog.relations and target != name:
            raise ValueError(
                f"Predicate guidance collision for {name!r}: an explicit reviewed "
                "catalog equivalence is required"
            )
        if target is not None and target in catalog.relations:
            relation = catalog.relations[target]
            if tuple(roles) != tuple(argument.name for argument in relation.arguments):
                raise ValueError(
                    f"Predicate guidance {name!r} -> {target!r} disagrees with "
                    "the catalog's arity or ordered argument roles"
                )
            mapped.append(
                f"- {target}({', '.join(roles)}) [catalog predicate; guidance "
                f"reused from {name}]: {description}"
            )
        else:
            provisional.append(f"- {name}({', '.join(roles)}): {description}")
    sections = []
    if mapped:
        sections.append("Reviewed catalog equivalents:\n" + "\n".join(mapped))
    if provisional:
        sections.append(
            "Uncataloged relation hints (use ProvisionalRelation):\n"
            + "\n".join(provisional)
        )
    if fallback:
        sections.append("Fallback interpretation guidance:\n" + "\n".join(fallback))
    return "\n\n".join(sections)


def _worked_examples(catalog: OntologyCatalog) -> str:
    selected_versions = set(catalog.versions)
    rendered = []
    for line_number, line in enumerate(
        _PROMPT_EXAMPLES_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            example = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid catalog example on line {line_number}"
            ) from error
        if not isinstance(example, dict) or set(example) not in (
            {"required_ontology_versions", "source_text", "entities", "assertions"},
            {
                "required_ontology_versions",
                "source_text",
                "entities",
                "assertions",
                "rules",
            },
        ):
            raise ValueError(f"Invalid catalog example fields on line {line_number}")
        required = example["required_ontology_versions"]
        if (
            not isinstance(required, list)
            or not required
            or any(not isinstance(version, str) or not version for version in required)
            or len(set(required)) != len(required)
            or not isinstance(example["source_text"], str)
            or not example["source_text"].strip()
            or not isinstance(example["entities"], list)
            or not isinstance(example["assertions"], list)
        ):
            raise ValueError(f"Invalid catalog example on line {line_number}")
        if not set(required) <= selected_versions:
            continue
        output = normalize_document_target(
            {
                "entities": example["entities"],
                "assertions": example["assertions"],
                "rules": example.get("rules", []),
            }
        )
        rendered.append(
            "<worked_example>\n"
            f"<example_source_text>\n{example['source_text']}\n</example_source_text>\n"
            f"<example_output_json>\n{json.dumps(output, ensure_ascii=False, separators=(',', ':'))}\n"
            "</example_output_json>\n</worked_example>"
        )
    if not rendered:
        raise ValueError("No catalog examples match the selected ontology versions")
    return "\n".join(rendered)


def build_catalog_document_prompt(
    source_text: str, *, catalog: OntologyCatalog, context: str = ""
) -> CatalogDocumentPrompt:
    """Render catalog guidance without modifying the source or its offsets."""

    if not source_text.strip():
        raise ValueError("source_text must contain non-whitespace text")

    raw = yaml.safe_load(_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Catalog prompt configuration must be a mapping")
    version = raw.get("prompt_version")
    system_prompt = raw.get("system_prompt")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Catalog prompt requires a nonempty prompt_version")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Catalog prompt requires a nonempty system_prompt")
    if (
        "{ontology_versions}" not in system_prompt
        or "{catalog_guide}" not in system_prompt
        or "{worked_examples}" not in system_prompt
        or "{extraction_guidance}" not in system_prompt
        or "{relation_guidance}" not in system_prompt
    ):
        raise ValueError("Catalog prompt is missing its ontology placeholders")

    guidance = (
        system_prompt.replace("{ontology_versions}", ", ".join(catalog.versions))
        .replace("{catalog_guide}", _catalog_guide(catalog))
        .replace("{extraction_guidance}", load_shared_extraction_guidance())
        .replace("{relation_guidance}", _relation_guidance(catalog))
        .replace("{worked_examples}", _worked_examples(catalog))
    )
    guidance = guidance.rstrip()
    # Source text varies per record; the extraction instructions must not vary
    # silently between annotation, training, and checkpoint inference.
    fingerprint = sha256(guidance.encode("utf-8")).hexdigest()
    return CatalogDocumentPrompt(
        version=f"{version}+sha256.{fingerprint}",
        text=(
            f"{guidance}\n\n"
            + (
                f"<reference_context>\n{context}\n</reference_context>\n"
                if context
                else ""
            )
            + f"<source_text>\n{source_text}\n</source_text>"
        ),
    )
