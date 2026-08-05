"""Internal KR metrics (research plan Appendix C, Stage A subset).

Takes the per-layer write matrices ``B_l`` (and optionally the write-node
outputs) and summarises how fractured / entangled the residual structure is.
Fully trace-safe under JAX transformations (jax.jit).
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphParams


@jax.jit
def fracture_index(
    B_l: jnp.ndarray,
    A_l: jnp.ndarray,
    groups: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Representation Fracture Index per semantic factor group.

    ``B_l`` is the write matrix ``(r, d)``, ``A_l`` the predictor read
    ``(d, r)`` (so the effective residual is ``B A h``). ``groups`` assigns each
    of the ``r`` residual components to a semantic factor. Returns the
    mean *between-group* to *within-group* variance ratio of the read
    directions ``A_l`` -- a low value means components carrying the same factor
    point in consistent directions (less fractured).
    """
    r = A_l.shape[-1]
    gids = jnp.asarray(groups, dtype=jnp.int32)

    def _calculate_index() -> jnp.ndarray:
        n_groups = jnp.max(gids) + 1

        # Normalize read directions per component: (r, d)
        dirs = A_l.T
        norms = jnp.linalg.norm(dirs, axis=-1, keepdims=True) + eps
        dirs = dirs / norms

        # Global centroid: (1, d)
        centroid = jnp.mean(dirs, axis=0, keepdims=True)

        # Count components per group: (n_groups, 1)
        group_counts = jax.ops.segment_sum(
            jnp.ones((r, 1)), gids, num_segments=n_groups
        )

        # Calculate group means (mu_g): (n_groups, d)
        group_sums = jax.ops.segment_sum(dirs, gids, num_segments=n_groups)
        mu_g = group_sums / (group_counts + eps)

        # Compute within-group variance
        dirs_mu = mu_g[gids]
        within = jnp.sum((dirs - dirs_mu) ** 2)

        # Compute between-group variance
        between = jnp.sum(group_counts * ((mu_g - centroid) ** 2))

        return between / (within + eps)

    # Clean guard conditions for r <= 0 or single group cases
    n_groups = jnp.max(gids) + 1 if gids.size > 0 else 0
    return jax.lax.cond(
        (r == 0) | (n_groups <= 1),
        lambda: jnp.array(0.0, dtype=A_l.dtype),
        _calculate_index,
    )


@jax.jit
def entanglement_index(
    B_l: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Entanglement Index: residual-direction overlap of the write matrix.

    ``B_l`` is ``(r, d)``. The index is the mean absolute cosine similarity
    between distinct write rows; 0 = orthogonal (independent residual
    directions), 1 = fully overlapping.
    """
    r = B_l.shape[0]

    def _calculate_index() -> jnp.ndarray:
        # L2-normalize rows of B_l: (r, d)
        rows = B_l / (jnp.linalg.norm(B_l, axis=-1, keepdims=True) + eps)

        # Compute Gram matrix (cosine similarities): (r, r)
        gram = jnp.abs(rows @ rows.T)

        # Zero out the diagonal self-similarity terms
        mask = 1.0 - jnp.eye(r, dtype=B_l.dtype)
        off_diag_gram = gram * mask

        # Average over all r * (r - 1) off-diagonal entries
        return jnp.sum(off_diag_gram) / (r * (r - 1))

    return jax.lax.cond(
        r <= 1,
        lambda: jnp.array(0.0, dtype=B_l.dtype),
        _calculate_index,
    )
