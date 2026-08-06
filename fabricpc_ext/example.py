import time
import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer

from fabricpc.graph_initialization import initialize_params
from fabricpc_ext import (
    BaseModelConfig,
    FrozenLLMExtension,
    PCResidualConfig,
    train_step,
)

print("1. Configuring frozen model and residual graph...")
base_config = BaseModelConfig(
    model_id="tiny-gpt2",
    layer_indices=(0,),
    max_seq_len=16,  # Short sequence length for this example
)

fabric_config = PCResidualConfig(
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

# Initialize parameters explicitly using FabricPC initializer
init_key = jax.random.PRNGKey(0)
params = initialize_params(graph, init_key)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

print("3. Preparing real text data (Shakespeare)...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

texts = [
    "To be, or not to be, that is the question:",
    "All the world's a stage, and all the men and women merely players;",
]

# Tokenize the text into integer IDs that the model understands
inputs = tokenizer(
    texts,
    return_tensors="np",
    max_length=base_config.max_seq_len,
    padding="max_length",
    truncation=True,
)
token_ids = jnp.array(inputs["input_ids"])
print(f"   Token IDs shape: {token_ids.shape}")

labels = jnp.zeros((len(texts), 10))

print("4. Caching frozen base activations...")
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