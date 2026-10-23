"""Public semantic-parser API; optional dependencies load on first use."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hybrid_miner.contracts import SemanticDocument

    from parser.semantic.backends import (
        AnthropicBackend,
        CallableBackend,
        GeminiBackend,
        ModelBackend,
        OllamaBackend,
        OpenAIBackend,
        OpenAICompatibleBackend,
        TransformersBackend,
    )
    from parser.semantic.dataset import (
        DistillationRecord,
        RejectedRecord,
        SemanticDatasetBuilder,
    )
    from parser.semantic.model_config import (
        ModelProfile,
        build_distilled_semantic_parser,
        build_reference_semantic_parser,
        create_backend,
        default_model_config_path,
        load_model_profile,
    )
    from parser.semantic.normalization import (
        SemanticNormalizationError,
        normalize_semantic_result,
        normalize_symbol,
    )
    from parser.semantic.schema import (
        SemanticArgument,
        SemanticAssertion,
        SemanticParseResult,
    )
    from parser.semantic.semantic_parser import (
        ALLOWED_PREDICATES,
        DistilledSemanticParser,
        ModelGenerationError,
        ReferenceSemanticParser,
        SemanticParseError,
        SemanticParserConfig,
    )

_EXPORTS = {
    "SemanticDocument": "hybrid_miner.contracts",
    **dict.fromkeys(
        (
            "AnthropicBackend",
            "CallableBackend",
            "GeminiBackend",
            "ModelBackend",
            "OllamaBackend",
            "OpenAIBackend",
            "OpenAICompatibleBackend",
            "TransformersBackend",
        ),
        "parser.semantic.backends",
    ),
    **dict.fromkeys(
        ("DistillationRecord", "RejectedRecord", "SemanticDatasetBuilder"),
        "parser.semantic.dataset",
    ),
    **dict.fromkeys(
        (
            "ModelProfile",
            "build_distilled_semantic_parser",
            "build_reference_semantic_parser",
            "create_backend",
            "default_model_config_path",
            "load_model_profile",
        ),
        "parser.semantic.model_config",
    ),
    **dict.fromkeys(
        ("SemanticNormalizationError", "normalize_semantic_result", "normalize_symbol"),
        "parser.semantic.normalization",
    ),
    **dict.fromkeys(
        ("SemanticArgument", "SemanticAssertion", "SemanticParseResult"),
        "parser.semantic.schema",
    ),
    **dict.fromkeys(
        (
            "ALLOWED_PREDICATES",
            "DistilledSemanticParser",
            "ModelGenerationError",
            "ReferenceSemanticParser",
            "SemanticParseError",
            "SemanticParserConfig",
        ),
        "parser.semantic.semantic_parser",
    ),
}
__all__ = [
    "ALLOWED_PREDICATES",
    "AnthropicBackend",
    "CallableBackend",
    "DistillationRecord",
    "DistilledSemanticParser",
    "GeminiBackend",
    "ModelBackend",
    "ModelGenerationError",
    "ModelProfile",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenAICompatibleBackend",
    "ReferenceSemanticParser",
    "RejectedRecord",
    "SemanticArgument",
    "SemanticAssertion",
    "SemanticDocument",
    "SemanticNormalizationError",
    "SemanticParseError",
    "SemanticParseResult",
    "SemanticParserConfig",
    "SemanticDatasetBuilder",
    "TransformersBackend",
    "build_distilled_semantic_parser",
    "build_reference_semantic_parser",
    "create_backend",
    "default_model_config_path",
    "load_model_profile",
    "normalize_semantic_result",
    "normalize_symbol",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    try:
        value = getattr(import_module(_EXPORTS[name]), name)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("hybrid_miner"):
            raise ModuleNotFoundError(
                "Semantic document parsing requires Hybrid Miner. "
                "Install the project with its [miner] extra on Python 3.11+."
            ) from error
        raise
    globals()[name] = value
    return value
