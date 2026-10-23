"""parser/semantic/model_config.py

Configuration-driven backend selection for semantic parsers.

This page contains:
  - ModelProfile:       one named model/provider configuration
  - load_model_profile: YAML → validated ModelProfile
  - create_backend:     ModelProfile → provider backend instance
  - build_reference_semantic_parser(): configuration → ready parser

The YAML file contains model choices. The local ``.env`` file contains API
keys. Keeping those responsibilities separate makes it safe to commit YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from parser.ontology.catalog import OntologyCatalog, load_ontology_catalog
from parser.semantic.backends import (
    AnthropicBackend,
    GeminiBackend,
    ModelBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenAICompatibleBackend,
    TransformersBackend,
)
from parser.semantic.semantic_parser import (
    DistilledSemanticParser,
    ReferenceSemanticParser,
    SemanticParserConfig,
)

# ── Model profile ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One named backend profile from the model configuration file."""

    name: str
    provider: str
    model: str
    settings: dict[str, Any]


# ── Configuration loading ────────────────────────────────────────────────────


def default_model_config_path() -> Path:
    """Return the repository's standard model-backend configuration path."""
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "parser_config"
        / "model_backend.yaml"
    )


def load_model_profile(
    config_path: str | Path | None = None,
    *,
    profile_name: str | None = None,
) -> ModelProfile:
    """Load the active (or explicitly named) model profile from YAML."""
    path = Path(config_path) if config_path is not None else default_model_config_path()
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Model configuration file was not found: {path}"
        ) from error
    if not isinstance(raw_config, dict):
        raise ValueError("Model configuration must be a YAML mapping")

    selected_name = profile_name or raw_config.get("active")
    profiles = raw_config.get("profiles")
    if not isinstance(selected_name, str) or not selected_name.strip():
        raise ValueError("Model configuration must define a non-empty 'active' profile")
    if not isinstance(profiles, dict):
        raise ValueError("Model configuration must define a 'profiles' mapping")
    raw_profile = profiles.get(selected_name)
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Unknown model profile: {selected_name!r}")

    provider = raw_profile.get("provider")
    model = raw_profile.get("model")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"Profile {selected_name!r} must define a provider")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Profile {selected_name!r} must define a model")
    return ModelProfile(
        name=selected_name,
        provider=provider.strip(),
        model=model.strip(),
        settings={
            key: value
            for key, value in raw_profile.items()
            if key not in {"provider", "model"}
        },
    )


# ── Environment ───────────────────────────────────────────────────────────────


def _load_environment() -> None:
    """Load local credentials without overwriting deployed environment values."""
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise ImportError(
            "Loading .env requires 'python-dotenv'. Install project dependencies first."
        ) from error
    load_dotenv(override=False)


def _required_environment(profile: ModelProfile, setting: str) -> str:
    variable_name = profile.settings.get(setting)
    if not isinstance(variable_name, str) or not variable_name:
        raise ValueError(f"Profile {profile.name!r} must define {setting!r}")
    value = os.getenv(variable_name)
    if not value:
        raise ValueError(
            f"Profile {profile.name!r} requires {variable_name}. "
            "Add it to .env or the process environment."
        )
    return value


# ── Backend factory ───────────────────────────────────────────────────────────


def create_backend(profile: ModelProfile) -> ModelBackend:
    """Instantiate the implementation selected by a model profile."""
    _load_environment()
    if profile.provider == "openai":
        _required_environment(profile, "api_key_env")
        return OpenAIBackend(
            max_output_tokens=int(profile.settings.get("max_output_tokens", 256))
        )
    if profile.provider == "anthropic":
        _required_environment(profile, "api_key_env")
        return AnthropicBackend(max_tokens=int(profile.settings.get("max_tokens", 256)))
    if profile.provider == "gemini":
        _required_environment(profile, "api_key_env")
        return GeminiBackend(
            max_output_tokens=int(profile.settings.get("max_output_tokens", 256))
        )
    if profile.provider == "ollama":
        think = profile.settings.get("think")
        if think is not None and not isinstance(think, bool):
            raise ValueError("Ollama profile 'think' must be a boolean")
        temperature = profile.settings.get("temperature")
        return OllamaBackend(
            max_output_tokens=int(profile.settings.get("max_output_tokens", 256)),
            think=think,
            temperature=float(temperature) if temperature is not None else None,
            seed=profile.settings.get("seed"),
        )
    if profile.provider == "openai-compatible":
        api_key_env = profile.settings.get("api_key_env")
        base_url_env = profile.settings.get("base_url_env")
        api_key = os.getenv(api_key_env) if isinstance(api_key_env, str) else None
        base_url = os.getenv(base_url_env) if isinstance(base_url_env, str) else None
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("OpenAI-compatible profiles require 'openai'.") from error
        return OpenAICompatibleBackend(
            OpenAI(api_key=api_key or "not-needed", base_url=base_url),
            max_output_tokens=int(profile.settings.get("max_output_tokens", 256)),
        )
    if profile.provider == "transformers-local":
        add_special_tokens = profile.settings.get("add_special_tokens", True)
        if not isinstance(add_special_tokens, bool):
            raise ValueError("Local profile 'add_special_tokens' must be a boolean")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("Local profiles require 'transformers'.") from error
        token_env = profile.settings.get("hf_token_env")
        token = os.getenv(token_env) if isinstance(token_env, str) else None
        tokenizer = AutoTokenizer.from_pretrained(profile.model, token=token)
        model = AutoModelForCausalLM.from_pretrained(profile.model, token=token)
        return TransformersBackend(
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=int(profile.settings.get("max_new_tokens", 256)),
            add_special_tokens=add_special_tokens,
        )
    raise ValueError(
        f"Unsupported provider {profile.provider!r} in profile {profile.name!r}"
    )


# ── Parser builder ────────────────────────────────────────────────────────────


def build_reference_semantic_parser(
    config_path: str | Path | None = None,
    *,
    profile_name: str | None = None,
    catalog: OntologyCatalog | None = None,
    model_revision: str | None = None,
    require_structured_output: bool = False,
) -> ReferenceSemanticParser:
    """Build a reference parser using the active YAML model profile."""

    profile = load_model_profile(config_path, profile_name=profile_name)
    if (
        profile.settings.get("role") == "student"
        or (Path(profile.model) / "training_metadata.json").is_file()
    ):
        from parser.semantic.catalog_student import validate_catalog_student_checkpoint

        validate_catalog_student_checkpoint(
            profile.model,
            catalog=catalog if catalog is not None else load_ontology_catalog(),
        )
    return ReferenceSemanticParser(
        backend=create_backend(profile),
        config=SemanticParserConfig(model_name=profile.model),
        catalog=catalog,
        model_revision=model_revision,
        require_structured_output=require_structured_output,
    )


def build_distilled_semantic_parser(
    config_path: str | Path | None = None,
    *,
    profile_name: str | None = None,
    catalog: OntologyCatalog | None = None,
) -> DistilledSemanticParser:
    """Build the configured local document-target student."""
    profile = load_model_profile(
        config_path, profile_name=profile_name or "semantic_document_student"
    )
    if (
        profile.provider != "transformers-local"
        or profile.settings.get("role") != "student"
    ):
        raise ValueError("Distilled parser requires a local student model profile")
    return DistilledSemanticParser.from_pretrained(
        model_dir=profile.model,
        device=str(profile.settings.get("device", "auto")),
        max_new_tokens=int(profile.settings.get("max_new_tokens", 3072)),
        catalog=catalog,
    )
