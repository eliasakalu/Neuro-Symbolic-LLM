"""Tier-1 frozen base: one JAX forward pass, per-layer hidden states cached.

This module owns the static substrate of the residual stack. It loads a
pretrained GPT-2-style causal LM via Hugging Face Flax (JAX-native), runs it
exactly once per batch, and returns the per-layer hidden states h_0^l as JAX
arrays. Those cached activations are the static reference points that the PC
residual fabric settles against, so the base is never re-run during the K
inference iterations ("activation caching").

The parameters theta_0 deliberately live outside the GraphParams tree so Optax
optimizers can never see or update them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
from transformers import GPT2Config, FlaxGPT2Model


@dataclass(frozen=True)
class BaseModelConfig:
    """Configuration of the frozen base model."""

    model_id: str = "tiny-gpt2"
    layer_indices: Tuple[int, ...] = (0,)
    vocab_size: int = 256
    max_seq_len: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 4

    @property
    def layer_keys(self) -> Tuple[str, ...]:
        return tuple(f"h{layer}" for layer in self.layer_indices)

    @property
    def is_tiny(self) -> bool:
        return self.model_id == "tiny-gpt2"


def _build_tiny_model(config: BaseModelConfig, prng_key: jax.Array) -> FlaxGPT2Model:
    """Construct a small random GPT-2 for offline JAX/Flax runs."""
    flax_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_len,
        n_ctx=config.max_seq_len,
        n_embd=config.hidden_size,
        n_layer=config.num_layers,
        n_head=config.num_heads,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    # Instantiate Flax model directly from config with JAX key initialization
    return FlaxGPT2Model(flax_config, seed=int(prng_key[0]))


class FrozenBase:
    """Frozen JAX/Flax GPT-2 substrate with a one-time forward activation cache."""

    def __init__(self, config: BaseModelConfig, prng_key: Optional[jax.Array] = None):
        self.config = config
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        if config.is_tiny:
            self.model = _build_tiny_model(config, prng_key)
        else:
            self.model = FlaxGPT2Model.from_pretrained(config.model_id)

    def hidden_states(self, token_ids: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Run the frozen base once using JAX and return cached hidden states.

        Args:
            token_ids: Integer token ids JAX array, shape ``(batch, seq_len)``.

        Returns:
            Dict mapping ``f"h{l}"`` (one per ``config.layer_indices`` entry)
            to the layer ``l`` hidden state, shape ``(batch, seq_len, d)``.
        """
        # Run Flax forward pass; output_hidden_states=True returns all intermediate layers
        outputs = self.model(input_ids=token_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple: [embeddings, h0, h1, ..., hL-1]

        result: Dict[str, jnp.ndarray] = {}
        for layer in self.config.layer_indices:
            # offset past token embedding output (index 0)
            result[f"h{layer}"] = hidden_states[layer + 1]

        return result


# ---------------------------------------------------------------------------
# Registry: theta_0 parameters live outside the JAX/Optax trainable tree.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, FrozenBase] = {}


def get_base(config: BaseModelConfig) -> FrozenBase:
    """Return the cached :class:`FrozenBase` for ``config`` (singleton)."""
    key = str(config)
    if key not in _REGISTRY:
        _REGISTRY[key] = FrozenBase(config)
    return _REGISTRY[key]
