# FabricPC Extension (`fabricpc_ext`)

> **Frozen Large Language Models meet Predictive Coding.**
>
> `fabricpc_ext` integrates a **completely frozen pretrained LLM** (e.g., GPT-2) with a **FabricPC continuous residual network**, allowing efficient downstream adaptation **without fine-tuning the base model** and **without backpropagating through the transformer**.

---

# Overview

Modern LLMs require enormous computational resources to fine-tune. This extension provides an alternative by keeping the pretrained transformer **100% frozen** while learning a lightweight **Predictive Coding (PC) residual fabric** that modifies the hidden representations locally.

Instead of updating billions of transformer parameters, the system learns a small residual graph that:

- caches hidden representations from the frozen model
- performs iterative predictive coding inference
- writes learned residuals back into the hidden states
- trains only the residual network and task head

The pretrained LLM never changes.

---

# Architecture

```
                  Input Tokens
                       │
                       ▼
         Frozen Pretrained LLM (GPT-2)
        (One Forward Pass • No Gradients)
                       │
             Cached Hidden States
                 h0[layer]
                       │
                       ▼
        FabricPC Continuous Residual Graph
      ┌────────────────────────────────────┐
      │ Identity Nodes                     │
      │ Predictive Coding Nodes            │
      │ Residual Write Nodes               │
      └────────────────────────────────────┘
                       │
          Iterative PC Inference
               (K inference steps)
                       │
                       ▼
        Residual Hidden Representation
               h̃[layer]
                       │
                       ▼
                Downstream Task
```

---

# Key Features

- ✅ Frozen pretrained transformer
- ✅ No backpropagation through the LLM
- ✅ Local Predictive Coding learning
- ✅ Low-rank residual adaptation
- ✅ Modular FabricPC implementation
- ✅ JAX + Flax implementation
- ✅ Hugging Face model support
- ✅ Lightweight transfer learning

---

# Learning Pipeline

The extension performs five stages during training.

## 1. Frozen Forward Pass

The pretrained language model performs **exactly one forward pass**.

For every selected transformer layer, the hidden representation is cached.

```
h0[layer]
```

These cached activations become the fixed inputs to the FabricPC graph.

The pretrained model parameters remain frozen throughout training.

---

## 2. Clamp Cached Activations

Each cached hidden state is connected to an Identity Node inside the FabricPC graph.

```
Cached Hidden State
        │
        ▼
 IdentityNode(h0_layer)
```

These values remain fixed during the iterative inference process.

---

## 3. Predictive Coding Inference

Instead of performing a conventional neural network forward pass, the residual graph performs **iterative inference**.

For every inference iteration:

- latent variables are updated
- prediction errors are minimized
- representations gradually converge

The predictive coding objective is

```text
Energy =
TaskLoss
+ (1 / (2σ²))
× || z[layer] − Predictor(h0[layer]) ||²
```

where

- `TaskLoss` is the downstream objective
- `Predictor()` predicts the latent state
- `z[layer]` is the latent PC representation

---

## 4. Residual Writing

After convergence, the learned latent representation is projected back into the frozen hidden state.

```text
ResidualHidden =
FrozenHidden
+
WriteMatrix × LatentState
```

or

```text
h̃[layer] = h0[layer] + B[layer] × z[layer]
```

This produces the updated hidden representation used for downstream prediction.

---

## 5. Local Learning

Only FabricPC parameters are updated.

Trainable components include

- predictor networks
- residual write matrices
- task heads

The pretrained transformer remains untouched.

---

# Directory Structure

```
fabricpc_ext/
│
├── __init__.py
├── env.py
├── metrics.py
├── base_forward.py
├── frozen_llm_extension.py
├── pc_residual.py
└── training.py
```

---

# File Descriptions

## base_forward.py

Responsible for the frozen transformer.

Responsibilities

- loads pretrained Hugging Face Flax models
- performs one forward pass
- caches hidden representations
- never computes gradients

Output

```
Hidden States
↓

h0[layer]
```

---

## frozen_llm_extension.py

Acts as the bridge between the frozen model and FabricPC.

Responsibilities

- owns the frozen model
- builds FabricPC graph
- converts token batches into clamp dictionaries
- connects cached activations to Identity Nodes

---

## pc_residual.py

Implements the complete Predictive Coding residual graph.

Contains

### IdentityNode

Stores cached hidden representations.

```
h0[layer]
```

---

### PCResidualNode

Represents the latent predictive coding state.

```
z[layer]
```

Responsibilities

- predictor network
- latent state update
- local predictive coding inference

---

### LinearResidual

Writes the learned residual back into the frozen hidden representation.

```
h̃[layer]
=
h0[layer]
+
B[layer] × z[layer]
```

---

## training.py

Contains the training algorithms.

### train_step()

Runs

```
Frozen Forward Pass
        │
        ▼
Clamp Hidden States
        │
        ▼
Predictive Coding Inference
        │
        ▼
Local Gradient Computation
        │
        ▼
Optimizer Update
```

---

### retrain_predictors()

Useful when changing the frozen pretrained model.

Only predictor networks are retrained.

Frozen

- residual write matrices
- task head
- pretrained transformer

Trainable

- predictor networks

---

## metrics.py

Provides Knowledge Representation metrics.

Includes

- Fracture Index
- Entanglement Index

These metrics measure how the residual network changes the internal representation.

---

## env.py

Configures

- JAX
- XLA
- runtime environment

before importing JAX.

---

# Iterative Predictive Coding

Unlike a standard transformer, FabricPC performs iterative inference.

```
Cached Hidden States
        │
        ▼
Inference Step 1
        │
        ▼
Inference Step 2
        │
        ▼
Inference Step 3
        │
        ▼
...
        │
        ▼
Inference Step K
        │
        ▼
Residual Output
```

Each iteration reduces the predictive coding energy until convergence.

---

# Example
> **Frozen Large Language Models meet Predictive Coding.**
>
> `fabricpc_ext` integrates a **completely frozen pretrained LLM** (e.g., GPT-2) with a **FabricPC continuous residual network**, allowing efficient downstream adaptation **without fine-tuning the base model** and **without backpropagating through the transformer**.

---

# Overview

Modern LLMs require enormous computational resources to fine-tune. This extension provides an alternative by keeping the pretrained transformer **100% frozen** while learning a lightweight **Predictive Coding (PC) residual fabric** that modifies the hidden representations locally.

Instead of updating billions of transformer parameters, the system learns a small residual graph that:

- caches hidden representations from the frozen model
- performs iterative predictive coding inference
- writes learned residuals back into the hidden states
- trains only the residual network and task head

The pretrained LLM never changes.

---

# Architecture

```
                  Input Tokens
                       │
                       ▼
         Frozen Pretrained LLM (GPT-2)
        (One Forward Pass • No Gradients)
                       │
             Cached Hidden States
                 h0[layer]
                       │
                       ▼
        FabricPC Continuous Residual Graph
      ┌────────────────────────────────────┐
      │ Identity Nodes                     │
      │ Predictive Coding Nodes            │
      │ Residual Write Nodes               │
      └────────────────────────────────────┘
                       │
          Iterative PC Inference
               (K inference steps)
                       │
                       ▼
        Residual Hidden Representation
               h̃[layer]
                       │
                       ▼
                Downstream Task
```

---

# Key Features

- ✅ Frozen pretrained transformer
- ✅ No backpropagation through the LLM
- ✅ Local Predictive Coding learning
- ✅ Low-rank residual adaptation
- ✅ Modular FabricPC implementation
- ✅ JAX + Flax implementation
- ✅ Hugging Face model support
- ✅ Lightweight transfer learning

---

# Learning Pipeline

The extension performs five stages during training.

## 1. Frozen Forward Pass

The pretrained language model performs **exactly one forward pass**.

For every selected transformer layer, the hidden representation is cached.

```
h0[layer]
```

These cached activations become the fixed inputs to the FabricPC graph.

The pretrained model parameters remain frozen throughout training.

---

## 2. Clamp Cached Activations

Each cached hidden state is connected to an Identity Node inside the FabricPC graph.

```
Cached Hidden State
        │
        ▼
 IdentityNode(h0_layer)
```

These values remain fixed during the iterative inference process.

---

## 3. Predictive Coding Inference

Instead of performing a conventional neural network forward pass, the residual graph performs **iterative inference**.

For every inference iteration:

- latent variables are updated
- prediction errors are minimized
- representations gradually converge

The predictive coding objective is

```text
Energy =
TaskLoss
+ (1 / (2σ²))
× || z[layer] − Predictor(h0[layer]) ||²
```

where

- `TaskLoss` is the downstream objective
- `Predictor()` predicts the latent state
- `z[layer]` is the latent PC representation

---

## 4. Residual Writing

After convergence, the learned latent representation is projected back into the frozen hidden state.

```text
ResidualHidden =
FrozenHidden
+
WriteMatrix × LatentState
```

or

```text
h̃[layer] = h0[layer] + B[layer] × z[layer]
```

This produces the updated hidden representation used for downstream prediction.

---

## 5. Local Learning

Only FabricPC parameters are updated.

Trainable components include

- predictor networks
- residual write matrices
- task heads

The pretrained transformer remains untouched.

---

# Directory Structure

```
fabricpc_ext/
│
├── __init__.py
├── env.py
├── metrics.py
├── base_forward.py
├── frozen_llm_extension.py
├── pc_residual.py
└── training.py
```

---

# File Descriptions

## base_forward.py

Responsible for the frozen transformer.

Responsibilities

- loads pretrained Hugging Face Flax models
- performs one forward pass
- caches hidden representations
- never computes gradients

Output

```
Hidden States
↓

h0[layer]
```

---

## frozen_llm_extension.py

Acts as the bridge between the frozen model and FabricPC.

Responsibilities

- owns the frozen model
- builds FabricPC graph
- converts token batches into clamp dictionaries
- connects cached activations to Identity Nodes

---

## pc_residual.py

Implements the complete Predictive Coding residual graph.

Contains

### IdentityNode

Stores cached hidden representations.

```
h0[layer]
```

---

### PCResidualNode

Represents the latent predictive coding state.

```
z[layer]
```

Responsibilities

- predictor network
- latent state update
- local predictive coding inference

---

### LinearResidual

Writes the learned residual back into the frozen hidden representation.

```
h̃[layer]
=
h0[layer]
+
B[layer] × z[layer]
```

---

## training.py

Contains the training algorithms.

### train_step()

Runs

```
Frozen Forward Pass
        │
        ▼
Clamp Hidden States
        │
        ▼
Predictive Coding Inference
        │
        ▼
Local Gradient Computation
        │
        ▼
Optimizer Update
```

---

### retrain_predictors()

Useful when changing the frozen pretrained model.

Only predictor networks are retrained.

Frozen

- residual write matrices
- task head
- pretrained transformer

Trainable

- predictor networks

---

## metrics.py

Provides Knowledge Representation metrics.

Includes

- Fracture Index
- Entanglement Index

These metrics measure how the residual network changes the internal representation.

---

## env.py

Configures

- JAX
- XLA
- runtime environment

before importing JAX.

---

# Iterative Predictive Coding

Unlike a standard transformer, FabricPC performs iterative inference.

```
Cached Hidden States
        │
        ▼
Inference Step 1
        │
        ▼
Inference Step 2
        │
        ▼
Inference Step 3
        │
        ▼
...
        │
        ▼
Inference Step K
        │
        ▼
Residual Output
```

Each iteration reduces the predictive coding energy until convergence.

---

# Example

```python
import jax
import jax.numpy as jnp
import optax
import time
from transformers import AutoTokenizer

from fabricpc_ext import (
    BaseModelConfig,
    PCResidualConfig,
    FrozenLLMExtension,
    train_step,
)

print("1. Configuring frozen model and residual graph...")
base_config = BaseModelConfig(
    model_id="tiny-gpt2",
    layer_indices=(0,),
    max_seq_len=16, # Short sequence length for this example
)

fabric_config = PCResidualConfig(
    seq_len=base_config.max_seq_len,
    seq_len=base_config.max_seq_len,
    hidden_size=base_config.hidden_size,
    num_classes=10,
    infer_steps=4,  # 4 iterations of PC inference
)

print("2. Loading the model and initializing FabricPC extension...")
start_time = time.time()
# This loads the HuggingFace model (or tiny-gpt2) and builds the residual graph
extension = FrozenLLMExtension(
    base_config,
    fabric_config,
)
print(f"   Done in {time.time() - start_time:.2f}s")

graph = extension.structure
params = graph.params

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

print("3. Preparing real text data (Shakespeare)...")
# Using a real tokenizer instead of random dummy integers!
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

texts = [
    "To be, or not to be, that is the question:",
    "All the world's a stage, and all the men and women merely players;"
]

# Tokenize the text into integer IDs that the model understands
inputs = tokenizer(
    texts, 
    return_tensors="np", 
    max_length=base_config.max_seq_len, 
    padding="max_length", 
    truncation=True
)
token_ids = jnp.array(inputs["input_ids"])
print(f"   Token IDs shape: {token_ids.shape}")

# Generate dummy labels (e.g., zero vectors for a classification task)
# In a real scenario, these would be your actual target labels!
labels = jnp.zeros((len(texts), 10))

print("4. Caching frozen base activations...")
# The base model runs exactly once here to cache activations and clamp them
batch = extension.build_task_batch(
    token_ids,
    labels,
)

print("5. Starting training loop (PC inference + local updates)...")
rng = jax.random.PRNGKey(42)
num_epochs = 5
for epoch in range(num_epochs):
    rng, step_rng = jax.random.split(rng)
    
    # train_step performs the K steps of PC inference internally and then updates weights
    params, opt_state, energy, state = train_step(
        params=params,
        opt_state=opt_state,
        batch=batch,
        structure=graph,
        optimizer=optimizer,
        rng_key=step_rng,
    )
    
    print(f"   Step {epoch + 1}/{num_epochs} - Predictive Energy: {energy:.4f}")

print("Training complete!")
```

---

# Advantages

Compared to conventional fine-tuning

| Feature | Full Fine-Tuning | FabricPC Extension |
|----------|------------------|--------------------|
| Updates Transformer | ✅ | ❌ |
| Requires Backprop Through LLM | ✅ | ❌ |
| Frozen Base Model | ❌ | ✅ |
| Local Learning | ❌ | ✅ |
| Lightweight Adaptation | ❌ | ✅ |
| Easy Transfer Learning | ⚠️ | ✅ |

---

# Requirements

- Python 3.10+
- JAX, Flax, Optax
- Hugging Face Transformers
- FabricPC

## Installation

Make sure you have `fabricpc` installed. If you haven't installed it, clone its repository and run the appropriate installation command for your system:

```bash
# GPU, CUDA 13
pip install -U -e ".[all,cuda13]"
# GPU, CUDA 12
pip install -U -e ".[all,cuda12]"
# CPU only
pip install -U -e ".[all]"
```

Install the required Hugging Face libraries:

```bash
pip install transformers flax
```

## Running the Example

If you run `python example.py` directly from inside the `fabricpc_ext/` directory, you will likely get a `ModuleNotFoundError: No module named 'fabricpc_ext'`. This happens because Python doesn't automatically add the parent directory to its path.

To run the example successfully, **execute it from the root of the repository** (e.g., inside the `Neuro-Symbolic-LLM` directory):

```bash
python -m fabricpc_ext.example
```
*or*
```bash
python fabricpc_ext/example.py
```

---

# Citation

If you use this extension in your research, please cite the corresponding FabricPC and Frozen-LLM publications.

---

# License

This module follows the same license as the parent **Neuro-Symbolic-LLM** project.