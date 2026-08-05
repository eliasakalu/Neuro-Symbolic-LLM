"""Frozen-LLM extension: ties the frozen JAX base to the PC residual fabric.

The base model weights ``theta_0`` never enter a ``fabricpc.core.types.GraphParams``
tree, so the Optax optimizer can never see or update them. Instead:

  * :class:`FrozenLLMExtension` owns the loaded :class:`FrozenBase` (Flax/JAX)
    and exposes the *Tier-1 activation cache*: one frozen forward pass per
    batch producing per-layer ``h0^l`` JAX arrays.
  * :meth:`FrozenLLMExtension.build_task_batch` turns a batch of token ids
    (and optional labels) into the clamp dictionary consumed by the FabricPC graph,
    mapping the cached arrays onto the ``h0_<l>`` input nodes via ``key_map``.

The base model is never re-run during the K PC inference iterations —
the cached activations are passed in once per batch before JIT.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp

from fabricpc_ext.base_forward import BaseModelConfig, FrozenBase, get_base
from fabricpc_ext.pc_residual import PCResidualConfig, build_pc_fabric


class FrozenLLMExtension:
    """Host-side bridge between the frozen base model and the PC residual graph.

    Args:
        base_config: Frozen base model configuration.
        fabric_config: PC residual fabric configuration. ``seq_len`` and
            ``hidden_size`` must match the base config / batched tokens.
    """

    def __init__(
        self,
        base_config: BaseModelConfig,
        fabric_config: Optional[PCResidualConfig] = None,
    ):
        self.base_config = base_config
        self.fabric_config = fabric_config or PCResidualConfig(
            seq_len=base_config.max_seq_len,
            hidden_size=base_config.hidden_size,
            layer_indices=base_config.layer_indices,
        )
        self.base = get_base(base_config)
        self.structure, self.key_map = build_pc_fabric(self.fabric_config)

    # -- Tier-1 activation cache -------------------------------------------

    def hidden_states(self, token_ids: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """One frozen base forward pass; returns cached ``h0^l`` JAX arrays."""
        return self.base.hidden_states(token_ids)

    # -- clamp dictionary for the FabricPC graph ---------------------------

    def build_task_batch(
        self,
        token_ids: jnp.ndarray,
        labels: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Assemble the batch dict consumed by the FabricPC training step.

        ``token_ids`` is run through the frozen base exactly once; the cached
        hidden states are clamped onto the ``h0_<l>`` input nodes. If ``labels``
        is provided and a task head is configured, it clamps the ``task`` output
        node, supplying ``L_task``.
        """
        h_all = self.base.hidden_states(token_ids)
        batch = {
            self.key_map[key]: arr
            for key, arr in h_all.items()
            if key in self.key_map
        }
        if labels is not None and "y" in self.key_map:
            batch[self.key_map["y"]] = jnp.asarray(labels)
        return batch
