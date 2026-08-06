"""Tier-1 frozen base: pure JAX/Flax forward pass to extract per-layer hidden states.

This module owns the static substrate of the residual stack. It runs a
GPT-2-style causal LM natively via Flax (JAX-native), runs it once per batch, and
returns per-layer hidden states h_0^l as JAX arrays. Those cached activations
are the static reference points that the PC residual fabric settles against,
so the base is never re-run during the K inference iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


# ---------------------------------------------------------------------------
# Native Flax GPT-2 Model (Pure JAX execution)
# ---------------------------------------------------------------------------


class FlaxGPT2Attention(nn.Module):
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        head_dim = self.hidden_size // self.num_heads
        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = nn.Dense(3 * self.hidden_size, name="c_attn")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention: (batch, num_heads, seq_len, head_dim)
        q = q.reshape((batch, seq_len, self.num_heads, head_dim)).swapaxes(1, 2)
        k = k.reshape((batch, seq_len, self.num_heads, head_dim)).swapaxes(1, 2)
        v = v.reshape((batch, seq_len, self.num_heads, head_dim)).swapaxes(1, 2)

        # Scaled dot-product attention
        attn_weights = jnp.matmul(q, k.swapaxes(-1, -2)) / jnp.sqrt(head_dim)
        if mask is not None:
            attn_weights = jnp.where(mask == 0, -1e9, attn_weights)
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_out = jnp.matmul(attn_probs, v)

        # Swap axes back and flatten heads
        attn_out = attn_out.swapaxes(1, 2).reshape((batch, seq_len, self.hidden_size))
        out = nn.Dense(self.hidden_size, name="c_proj")(attn_out)
        return out


class FlaxGPT2MLP(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(4 * self.hidden_size, name="c_fc")(x)
        h = jax.nn.gelu(h, approximate=True)
        out = nn.Dense(self.hidden_size, name="c_proj")(h)
        return out


class FlaxGPT2Block(nn.Module):
    hidden_size: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        ln_1 = nn.LayerNorm(epsilon=1e-5, name="ln_1")(x)
        attn_out = FlaxGPT2Attention(self.hidden_size, self.num_heads, name="attn")(ln_1, mask=mask)
        x = x + attn_out

        ln_2 = nn.LayerNorm(epsilon=1e-5, name="ln_2")(x)
        mlp_out = FlaxGPT2MLP(self.hidden_size, name="mlp")(ln_2)
        x = x + mlp_out
        return x


class FlaxGPT2Model(nn.Module):
    vocab_size: int
    max_seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int

    @nn.compact
    def __call__(
        self, input_ids: jnp.ndarray, output_hidden_states: bool = True
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, ...]]]:
        batch, seq_len = input_ids.shape
        pos_ids = jnp.arange(0, seq_len)[None, :]

        wte = nn.Embed(self.vocab_size, self.hidden_size, name="wte")(input_ids)
        wpe = nn.Embed(self.max_seq_len, self.hidden_size, name="wpe")(pos_ids)
        x = wte + wpe

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]

        all_hidden_states = [x] if output_hidden_states else None

        for i in range(self.num_layers):
            block = FlaxGPT2Block(self.hidden_size, self.num_heads, name=f"h_{i}")
            x = block(x, mask=causal_mask)
            if output_hidden_states:
                all_hidden_states.append(x)

        x = nn.LayerNorm(epsilon=1e-5, name="ln_f")(x)

        return x, tuple(all_hidden_states) if all_hidden_states is not None else None


# ---------------------------------------------------------------------------
# BaseModelConfig & FrozenBase Substrate
# ---------------------------------------------------------------------------


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


class FrozenBase:
    """Frozen pure JAX/Flax GPT-2 substrate with a one-time forward activation cache."""

    def __init__(self, config: BaseModelConfig, prng_key: Optional[jax.Array] = None):
        self.config = config
        prng_key = prng_key if prng_key is not None else jax.random.PRNGKey(0)

        self.model = FlaxGPT2Model(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )
        dummy_input = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
        self.params = self.model.init(prng_key, dummy_input)

    def hidden_states(self, token_ids: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Run the frozen base once using pure JAX and return cached hidden states.

        Args:
            token_ids: Integer token ids JAX array, shape ``(batch, seq_len)``.

        Returns:
            Dict mapping ``f"h{l}"`` (one per ``config.layer_indices`` entry)
            to the layer ``l`` hidden state, shape ``(batch, seq_len, d)``.
        """
        _, all_hidden_states = self.model.apply(
            self.params, token_ids, output_hidden_states=True
        )
        if all_hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")

        result: Dict[str, jnp.ndarray] = {}
        for layer in self.config.layer_indices:
            # offset past token embedding output (index 0)
            result[f"h{layer}"] = all_hidden_states[layer + 1]

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


