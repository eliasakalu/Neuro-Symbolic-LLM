"""Internal KR metrics (research plan Appendix C, Stage A subset).

Takes the per-layer write matrices ``B_l`` (and optionally the write-node
outputs) and summarises how fractured / entangled the residual structure is.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from fabricpc.core.types import GraphParams


def fracture_index(
    B_l: jnp.ndarray,
    A_l: jnp.ndarray,
    groups: jnp.ndarray,
    eps: float = 1e-8,
) -> float:
    """Representation Fracture Index per semantic factor group.

    ``B_l`` is the write matrix ``(r, d)``, ``A_l`` the predictor read
    ``(d, r)`` (so the effective residual is ``B A h``). ``groups`` assigns each
    of the ``r`` residual components to a semantic factor. Returns the
    mean *between-group* to *within-group* variance ratio of the read
    directions ``A_l`` -- a low value means components carrying the same factor
    point in consistent directions (less fractured).
    """
    r = A_l.shape[-1]
    if r == 0:
        return 0.0
    gids = jnp.asarray(groups)
    n_groups = int(gids.max()) + 1
    if n_groups <= 1:
        return 0.0

    dirs = A_l.T  # (r, d), normalised per component
    norms = jnp.linalg.norm(dirs, axis=-1, keepdims=True) + eps
    dirs = dirs / norms

    within, between = 0.0, 0.0
    centroid = dirs.mean(axis=0, keepdims=True)
    for g in range(n_groups):
        idx = jnp.where(gids == g)[0]
        if idx.shape[0] == 0:
            continue
        comps = dirs[idx]
        mu_g = comps.mean(axis=0, keepdims=True)
        within += jnp.sum((comps - mu_g) ** 2)
        between += comps.shape[0] * jnp.sum((mu_g - centroid) ** 2)
    return float(between / (within + eps))


def entanglement_index(
    B_l: jnp.ndarray,
    eps: float = 1e-8,
) -> float:
    """Entanglement Index: residual-direction overlap of the write matrix.

    ``B_l`` is ``(r, d)``. The index is the mean absolute cosine similarity
    between distinct write rows; 0 = orthogonal (independent residual
    directions), 1 = fully overlapping.
    """
    rows = B_l / (jnp.linalg.norm(B_l, axis=-1, keepdims=True) + eps)
    gram = rows @ rows.T
    r = gram.shape[0]
    off = jnp.abs(gram) - jnp.eye(r)
    if r <= 1:
        return 0.0
    return float(jnp.sum(off) / (r * (r - 1)))
