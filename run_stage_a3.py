import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import numpy as np
import optax
import jax

from fabricpc_ext import (
    FrozenLLMExtension,
    BaseModelConfig,
    PCResidualConfig,
    train_step,
)
from fabricpc.graph_initialization import initialize_params

cfg = BaseModelConfig(
    model_id="tiny-gpt2",
    max_seq_len=8,
    vocab_size=128,
    hidden_size=32,
    num_layers=2,
    num_heads=4,
)
ext = FrozenLLMExtension(
    cfg,
    PCResidualConfig(
        seq_len=8,
        hidden_size=32,
        residual_rank=4,
        layer_indices=(0, 1),
        num_classes=10,
        infer_steps=4,
        eta_infer=0.1,
    ),
)
params = initialize_params(ext.structure, jax.random.PRNGKey(0))
opt = optax.adam(1e-3)
opt_state = opt.init(params)

tokens = np.random.randint(0, 128, (4, 8))
labels = np.random.randint(0, 10, (4, 1))
batch = ext.build_task_batch(tokens, labels)

params, opt_state, energy, _ = train_step(
    params, opt_state, batch, ext.structure, opt, jax.random.PRNGKey(1)
)
print("energy:", energy)