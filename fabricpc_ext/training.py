"""Training loop for the PC residual fabric.

Wraps the stock FabricPC machinery (``initialize_graph_state`` +
``run_inference`` + ``compute_local_weight_gradients``) into a single
``train_step``. The frozen base never enters the Optax tree: ``theta_0`` lives
in the host-side :class:`FrozenBase`, and only the fabric/task weights live in
``GraphParams``.

Also provides *predictor-only retraining* (research plan transfer section): for
a changed base, freeze the write matrices ``B_l`` (and task head), fix the
latent targets ``z_l`` from a reference run, and retrain only the predictors
``f_phi`` against the new base activations.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state


def get_graph_param_gradient(
    params: GraphParams,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    rng_key: jax.Array,
) -> Tuple[GraphParams, float, GraphState]:
    """Local weight gradients for a batch (single device, JIT-able).

    Mirrors ``fabricpc.training.train.get_graph_param_gradient``, but every
    batch value is already a clamp (cached ``h0^l`` arrays + labels), so the
    task-map lookup always fires and the base is never re-run.
    """
    batch_size = next(iter(batch.values())).shape[0]
    clamps = {structure.task_map[k]: v for k, v in batch.items() if k in structure.task_map}

    init_state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )
    final_state = run_inference(params, init_state, clamps, structure)

    energy = sum(
        sum(final_state.nodes[node_name].energy)
        for node_name in structure.nodes
        if structure.nodes[node_name].node_info.in_degree > 0
    ) / batch_size

    grads = compute_local_weight_gradients(params, final_state, structure)
    return grads, float(energy), final_state


def train_step(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    *,
    mask: Optional[Callable] = None,
) -> Tuple[GraphParams, optax.OptState, float, GraphState]:
    """One PC training step: infer -> local gradients -> optimizer update.

    Args:
        mask: Optional ``optax``-style gradient mask (e.g. keep only the
            predictor weights for predictor-only retraining).
    """
    grads, energy, final_state = get_graph_param_gradient(
        params, batch, structure, rng_key
    )
    if mask is not None:
        grads = _apply_mask(grads, mask)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, energy, final_state


def _apply_mask(grads: GraphParams, mask: Callable) -> GraphParams:
    """Zero gradients on nodes the mask function rejects."""

    def per_node(name, node_params):
        keep = mask(name)
        weights = {k: (v if keep else jnp.zeros_like(v)) for k, v in node_params.weights.items()}
        biases = {k: (v if keep else jnp.zeros_like(v)) for k, v in node_params.biases.items()}
        return type(node_params)(weights=weights, biases=biases)

    nodes = {
        name: per_node(name, np)
        for name, np in grads.nodes.items()
    }
    return type(grads)(nodes=nodes)


def predictor_only_mask(node_name: str) -> bool:
    """Gradient mask that keeps only predictor weights (``z_<l>`` nodes)."""
    return node_name.startswith("z_")


def retrain_predictors(
    params: GraphParams,
    structure: GraphStructure,
    new_base_batches: "object",
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    num_steps: int = 1,
) -> GraphParams:
    """Predictor-only retraining against a changed base.

    ``B_l`` and the task head are frozen (via :func:`predictor_only_mask`);
    only the ``f_phi`` predictors in the ``z_<l>`` nodes update, driving their
    latents toward the cached activations of the new base. This is the
    research-plan transfer trick: PC separates state inference from weight
    learning, so the residual's learned latent targets can stay fixed while the
    predictor re-learns the read direction.

    Args:
        params: Current fabric parameters.
        structure: Fabric graph.
        new_base_batches: Iterable of batch dicts (cached new-base
            activations, same clamp format as ``train_step``).
        optimizer: Optax optimizer (e.g. ``optax.adam(1e-3)``).
        rng_key: JAX PRNG key, split per step.
        num_steps: Number of predictor-only update steps.

    Returns:
        Updated ``GraphParams`` with fresh predictor weights.
    """
    opt_state = optimizer.init(params)
    for batch in new_base_batches:
        key, rng_key = jax.random.split(rng_key)
        params, opt_state, _, _ = train_step(
            params, opt_state, batch, structure, optimizer, key,
            mask=predictor_only_mask,
        )
        num_steps -= 1
        if num_steps <= 0:
            break
    return params