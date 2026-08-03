"""Tier-1 frozen base: one torch forward pass, per-layer hidden states cached.

This module owns the *static substrate* of the residual stack. It loads a
pretrained GPT-2-style causal LM (HuggingFace ``transformers``, PyTorch), runs
it exactly once per batch under ``torch.no_grad()``, and returns the per-layer
hidden states ``h_0^l`` as JAX arrays. Those cached activations are the static
reference points that the PC residual fabric settles against, so the base is
never re-run during the K inference iterations ("activation caching").

The weights ``theta_0`` deliberately live here — in a Python-level registry —
and never inside a ``fabricpc.core.types.GraphParams`` tree, so the Optax
optimizer can never see or update them. That is the freeze mechanism.

Example:
    >>> from fabricpc_ext.base_forward import FrozenBase
    >>> base = FrozenBase(tiny_config(model_id="tiny-gpt2"))
    >>> h_all = base.hidden_states(tokens)   # dict: layer_index -> jnp array
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# torch is imported lazily so that importing this package never requires a
# torch build; the frozen base is the only consumer of PyTorch.
_torch = None


@dataclass(frozen=True)
class BaseModelConfig:
    """Configuration of the frozen base model.

    Attributes:
        model_id: A HuggingFace hub id (e.g. ``"gpt2"``) or the reserved
            sentinel ``"tiny-gpt2"`` which builds a small random GPT-2 from a
            config (offline, for tests and CPU smoke runs).
        layer_indices: Base transformer layers whose hidden states the residual
            fabric attaches to (0-based transformer layer ids). Hidden state
            ``l`` is the output of transformer layer ``l``; the token embedding
            output is index -1 and is never exposed.
        vocab_size: Model vocabulary size (used for the tiny sentinel only).
        max_seq_len: Maximum sequence length / position ids.
        hidden_size: Transformer embedding dimension.
        num_layers: Transformer depth.
        num_heads: Attention heads.
    """

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


def _load_torch() -> "Any":
    """Import torch on first use (avoids import cost when unused)."""
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _build_tiny_model(config: BaseModelConfig):
    """Construct a small random GPT-2 for offline runs."""
    torch = _load_torch()
    from transformers import GPT2Config, GPT2Model

    gpt2_config = GPT2Config(
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
    model = GPT2Model(gpt2_config)
    for param in model.parameters():
        with torch.no_grad():
            param.normal_(std=0.02)
    return model


class FrozenBase:
    """Frozen GPT-2 substrate with a one-time-forward activation cache.

    The instance holds the torch model in evaluation mode. Each call to
    :meth:`hidden_states` is exactly one forward pass; the returned dictionary
    of JAX arrays is the Tier-1 activation cache consumed by the residual
    fabric graph.
    """

    def __init__(self, config: BaseModelConfig, device: Optional[str] = None):
        self.config = config
        torch = _load_torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if config.is_tiny:
            self.model = _build_tiny_model(config)
        else:
            from transformers import GPT2Model

            self.model = GPT2Model.from_pretrained(config.model_id)
        self.model.to(self.device)
        self.model.eval()

    def hidden_states(
        self, token_ids: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Run the frozen base once and return cached hidden states.

        Args:
            token_ids: Integer token ids, shape ``(batch, seq_len)``.

        Returns:
            Dict mapping ``f"h{l}"`` (one per ``config.layer_indices`` entry)
            to the layer ``l`` hidden state, shape ``(batch, seq_len, d)``.
        """
        torch = _load_torch()
        tokens = torch.as_tensor(np.asarray(token_ids), device=self.device)
        with torch.no_grad():
            outputs = self.model(tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # [embed, h0, h1, ..., hL-1]
        result: Dict[str, jnp.ndarray] = {}
        for layer in self.config.layer_indices:
            tensor = hidden_states[layer + 1]  # offset past the embedding slot
            result[f"h{layer}"] = jnp.asarray(
                tensor.detach().to("cpu").float().numpy()
            )
        return result


# ---------------------------------------------------------------------------
# Registry: theta_0 lives outside the JAX/Optax trainable tree.
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, FrozenBase] = {}


def get_base(config: BaseModelConfig) -> FrozenBase:
    """Return the cached :class:`FrozenBase` for ``config`` (one instance).

    The base model weights are registered here, keyed by the frozen config, so
    multiple ``FrozenLLMExtension`` graph nodes can share a single loaded
    substrate without duplicating weights.
    """
    key = str(config)
    if key not in _REGISTRY:
        _REGISTRY[key] = FrozenBase(config)
    return _REGISTRY[key]
