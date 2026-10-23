"""Tests for configuration-driven semantic-parser backend selection."""

import yaml

from parser.semantic.model_config import default_model_config_path, load_model_profile


def test_loads_active_profile_from_project_config() -> None:
    config = yaml.safe_load(default_model_config_path().read_text(encoding="utf-8"))
    profile = load_model_profile()

    assert profile.name == config["active"]
    assert profile.provider == config["profiles"][profile.name]["provider"]
    assert profile.model == config["profiles"][profile.name]["model"]


def test_can_select_named_profile() -> None:
    profile = load_model_profile(profile_name="ollama_qwen3_8b")

    assert profile.name == "ollama_qwen3_8b"
    assert profile.provider == "ollama"
    assert profile.model


def test_catalog_ollama_profile_is_explicit_and_legacy_profile_is_unchanged() -> None:
    legacy = load_model_profile(profile_name="ollama")
    catalog = load_model_profile(profile_name="ollama_catalog")

    assert legacy.model == catalog.model
    assert "think" not in legacy.settings
    assert "temperature" not in legacy.settings
    assert catalog.settings["think"] is False
    assert catalog.settings["temperature"] == 0.0


def test_distilled_factory_uses_shipped_provider_budget_and_catalog(monkeypatch):
    from dataclasses import replace

    from parser.ontology.catalog import load_ontology_catalog
    from parser.semantic import model_config

    profile = load_model_profile(profile_name="semantic_document_student")
    profile = replace(profile, settings={**profile.settings, "max_new_tokens": 777})
    monkeypatch.setattr(model_config, "load_model_profile", lambda *_, **__: profile)
    calls = []
    monkeypatch.setattr(
        model_config.DistilledSemanticParser,
        "from_pretrained",
        lambda **kwargs: calls.append(kwargs),
    )
    catalog = load_ontology_catalog(modules=("upper",))
    model_config.build_distilled_semantic_parser(catalog=catalog)
    assert calls[0]["max_new_tokens"] == 777
    assert calls[0]["catalog"] is catalog
