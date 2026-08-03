"""Frozen-LLM + FabricPC continuous residual stack (research plan Stage A3).

Public API:
    - ``setup_env``        : configure JAX/XLA before importing JAX.
    - ``FrozenLLMExtension``: host-side bridge (frozen torch base + graph).
    - ``BaseModelConfig``  : frozen base config (``"tiny-gpt2"`` = offline).
    - ``PCResidualConfig`` / ``build_pc_fabric``: build the residual graph.
    - ``train_step`` / ``retrain_predictors``: PC training + transfer.
    - ``fracture_index`` / ``entanglement_index``: internal KR metrics.
"""

from fabricpc_ext.env import setup_env

setup_env()

from fabricpc_ext.base_forward import BaseModelConfig, FrozenBase, get_base
from fabricpc_ext.frozen_llm_extension import FrozenLLMExtension
from fabricpc_ext.pc_residual import PCResidualConfig, PCResidualNode, build_pc_fabric
from fabricpc_ext.training import (
    train_step,
    get_graph_param_gradient,
    retrain_predictors,
    predictor_only_mask,
)
from fabricpc_ext.metrics import fracture_index, entanglement_index

__all__ = [
    "setup_env",
    "BaseModelConfig",
    "FrozenBase",
    "get_base",
    "FrozenLLMExtension",
    "PCResidualConfig",
    "PCResidualNode",
    "build_pc_fabric",
    "train_step",
    "get_graph_param_gradient",
    "retrain_predictors",
    "predictor_only_mask",
    "fracture_index",
    "entanglement_index",
]