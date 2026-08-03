"""Environment setup for the Frozen-LLM + FabricPC residual stack.

Call :func:`setup_env` before importing JAX anywhere in the process. It is a
no-op after the first call and imports are guarded so a second call is safe.
"""

from __future__ import annotations

import os

_XLA_FRACTION = "0.9"


def setup_env() -> None:
    """Configure the runtime environment for the Tier-1 dense GPU loop.

    * ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`` reserves device memory headroom
      for the PC latent-state buffers (research plan, Point 3: batch sizes
      tuned for ~50% device memory headroom).
    * ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is left off by design: the
      FabricPC tier preallocates on its own schedule; overriding it here would
      fight the JAX default for no benefit on this stack.

    Safe to call multiple times; only sets variables that are not already set.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", _XLA_FRACTION)
