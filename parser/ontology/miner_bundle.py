"""Compile a versioned ontology catalog into the miner's bundle contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from hybrid_miner.ontology.bundle import OntologyBundle

from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog

DEFAULT_BUNDLE_MANIFEST = (
    Path(__file__).resolve().parent / "bundles" / "semantic-extraction-v1.yaml"
)


def load_catalog_bundle(
    manifest_path: Path = DEFAULT_BUNDLE_MANIFEST,
    *,
    ontology_directory: Path | None = None,
) -> tuple[OntologyCatalog, OntologyBundle]:
    """Load a declared module set and reject module-version mismatches."""

    try:
        raw: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"Cannot load ontology bundle manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(raw, dict) or set(raw) != {
        "bundle_id",
        "bundle_version",
        "modules",
    }:
        raise ValueError(
            "Ontology bundle manifest requires bundle_id, bundle_version, and modules"
        )
    modules = raw["modules"]
    if (
        not isinstance(modules, dict)
        or not modules
        or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not version.strip()
            for name, version in modules.items()
        )
    ):
        raise ValueError(
            "Ontology bundle modules must map names to version identifiers"
        )
    catalog = load_ontology_catalog(ontology_directory, modules=modules)
    actual_versions = dict(zip(sorted(modules), catalog.versions, strict=True))
    if actual_versions != modules:
        raise ValueError(
            f"Ontology bundle version mismatch: declared={modules}, actual={actual_versions}"
        )
    bundle = OntologyBundle.model_validate(
        {
            "contract_version": "ontology-bundle-v1",
            "bundle_id": raw["bundle_id"],
            "bundle_version": raw["bundle_version"],
            "ontology_versions": catalog.versions,
            "types": sorted(catalog.types | {"String"}),
            "predicates": [
                {
                    "name": relation.name,
                    "ontology_version": relation.ontology_version,
                    "argument_types": [
                        list(argument.allowed_types) for argument in relation.arguments
                    ],
                }
                for relation in sorted(
                    catalog.relations.values(), key=lambda item: item.name
                )
            ],
        }
    )
    return catalog, bundle
