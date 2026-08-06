# FabricPC Extension (`fabricpc_ext`)

This folder contains the **Frozen-LLM + FabricPC Continuous Residual Stack**, implementing the integration of a completely frozen Large Language Model (LLM) base with a Predictive Coding (PC) residual fabric (FabricPC).

## Overview
The architecture is designed to augment a frozen pretrained LLM (like GPT-2) with a continuous, low-rank residual graph trained via Predictive Coding. This allows for adapting and tuning the model representations for downstream tasks without ever modifying the base model's weights or requiring backpropagation through time/layers of the massive LLM.

## How it Works
1. **Activation Caching (One-Time Forward)**: The Frozen LLM (substrate) receives an input batch of tokens and runs exactly one forward pass using JAX/Flax. The hidden states $h_0^l$ for selected layers are cached. The base model parameters ($\theta_0$) never enter the trainable Optax tree.
2. **Clamping to the Fabric**: The cached hidden states are mapped and clamped onto identity input nodes ($h0\_<l>$) in the FabricPC residual graph.
3. **Iterative Inference (PC)**: For $K$ inference steps, the predictive coding fabric settles its continuous latent variables ($z_l$) by minimizing a Gaussian predictive energy function:
   $$E = L_{task} + \frac{1}{2\sigma^2} || z_l - f_\phi(h_0^l) ||^2$$
   Here, $f_\phi$ is a learned predictor predicting the latent state from the frozen hidden state.
4. **Residual Write**: The settled latent state is transformed by a low-rank write matrix ($B_l$) and added back to the frozen hidden state to produce the final representation: $\tilde{h}^l = h_0^l + B_l z_l$.
5. **Local Learning**: Weight gradients for the FabricPC nodes ($f_\phi$, $B_l$, and task heads) are computed locally based on the settled PC state, without backpropagating through the frozen base.

## Components and Files

### `base_forward.py`
**Purpose**: Hosts the static substrate of the residual stack.
- Loads a pretrained GPT-2-style causal LM via Hugging Face Flax.
- Defines `FrozenBase`, which runs one forward pass per batch and caches per-layer hidden states ($h_0^l$) as JAX arrays. 
- These cached activations serve as static reference points for the PC residual fabric.

### `frozen_llm_extension.py`
**Purpose**: The host-side bridge that ties the frozen JAX base to the PC residual graph.
- Defines `FrozenLLMExtension`, which owns the `FrozenBase`.
- Implements `build_task_batch()`, which turns a batch of token IDs into a clamp dictionary consumed by the FabricPC graph, mapping the cached arrays onto the $h0\_<l>$ input nodes.

### `pc_residual.py`
**Purpose**: Defines the continuous PC residual fabric built on FabricPC node primitives.
- **`IdentityNode` ($h0\_<l>$)**: Clamped to the cached frozen base hidden state $h_0^l$.
- **`PCResidualNode` ($z\_<l>$)**: Represents the latent state $z_l$. Its predictor $f_\phi$ reads $h_0^l$. It minimizes local predictive energy. The $K$ PC inference iterations dynamically update its latent state.
- **`LinearResidual` ($out\_<l>$)**: The write node that computes $\tilde{h}^l = B_l z_l + h_0^l$.
- Constructs the full graph structure and maps the topology via `build_pc_fabric()`.

### `training.py`
**Purpose**: Handles the training loops and parameter updates for the FabricPC graph.
- **`train_step`**: Wraps the stock FabricPC machinery (inference $\rightarrow$ local gradients $\rightarrow$ optimizer update).
- **`retrain_predictors` (Predictor-only retraining)**: Facilitates transfer learning when the base model is changed. It freezes the write matrices ($B_l$) and task head, and retrains *only* the predictors ($f_\phi$) in the $z\_<l>$ nodes against the new base activations.

### `metrics.py`
**Purpose**: Contains internal Knowledge Representation (KR) metrics, specifically computing the fracture index and entanglement index to analyze the residual modifications.

### `env.py`
**Purpose**: Environment setup to configure JAX/XLA backend settings prior to importing JAX.

### `__init__.py`
**Purpose**: Exposes the public API for the module.

## Iterative Process (PC Inference)
In a standard deep learning model, a forward pass is instantaneous. In this FabricPC implementation, the forward pass involves an **iterative relaxation process**:
1. The frozen hidden states $h_0^l$ are clamped.
2. The network runs for `infer_steps` (e.g., $K=4$).
3. At each step, the latent variables $z_l$ dynamically update themselves by following the gradient of the local energy functional (trying to match the prediction $f_\phi(h_0^l)$ and any top-down task signals).
4. Once the latent states settle, the final residual $\tilde{h}^l$ is computed and passed to the task head.

## How to Run / Use
You can utilize the `FrozenLLMExtension` to bridge a model and train it using standard JAX/Optax paradigms wrapped by FabricPC. 

```python
import jax
import optax
from fabricpc_ext import BaseModelConfig, PCResidualConfig, FrozenLLMExtension, train_step

# 1. Configure the frozen base and the PC residual fabric
base_config = BaseModelConfig(model_id="tiny-gpt2", layer_indices=(0,))
fabric_config = PCResidualConfig(
    seq_len=base_config.max_seq_len, 
    hidden_size=base_config.hidden_size,
    num_classes=10
)

# 2. Initialize the extension (loads model & builds graph)
extension = FrozenLLMExtension(base_config, fabric_config)
graph_structure = extension.structure
params = graph_structure.params # Initialized node parameters

# 3. Setup Optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# 4. Training Loop
# Assuming `token_ids` and `labels` are your JAX arrays for a batch
batch_dict = extension.build_task_batch(token_ids, labels)
rng_key = jax.random.PRNGKey(42)

# Run one train step (Inference + Local Weight Update)
params, opt_state, energy, state = train_step(
    params=params,
    opt_state=opt_state,
    batch=batch_dict,
    structure=graph_structure,
    optimizer=optimizer,
    rng_key=rng_key
)
print(f"Energy: {energy}")
```
