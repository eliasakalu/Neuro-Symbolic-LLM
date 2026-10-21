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
