"""Continuous PC residual fabric (Stage A3) built on FabricPC node primitives.

The fabric realises, for each residual layer ``l``:

    h~l = h0l + B_l z_l,        z_l approx equal to f_phi(h0l),

with the Gaussian predictive energy of the research plan, Appendix A:

    E = L_task + (1 / 2 sigma^2) || z_l - f_phi(h0l) ||^2.

Three FabricPC nodes are composed per layer:

  * ``h0_<l>`` :class:`IdentityNode` -- clamped to the *cached* frozen base
    hidden state ``h0l`` (produced host-side by one torch forward per batch).
    It is the static reference the residual settles against.
  * ``z_<l>``   :class:`PCResidualNode` -- the latent state ``z_l`` that the K
    PC inference iterations move. Its predictor ``f_phi`` reads ``h0l``; its
    Gaussian energy is exactly the local predictive term ``||z_l - f_phi(h0l)||^2``.
  * ``out_<l>`` :class:`LinearResidual` -- the write node ``z_mu = B_l z_l + h0l``
    (weighted ``in`` slot from ``z_<l>``, identity ``skip`` slot from ``h0_<l>``).

An optional task readout (``AvgPool`` -> ``Linear``) closes the loop and
supplies ``L_task``; its latent is clamped to the labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, NormalInitializer
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear import Linear
from fabricpc.nodes.linear_residual import LinearResidual
from fabricpc.nodes.pooling import AvgPool
from fabricpc.graph_assembly.graph_construction import graph, TaskMap
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit


class PCResidualNode(NodeBase):
    """Latent residual node.

    Predicts its own latent ``z`` from the frozen base hidden state ``h0``:

        z_mu = f_phi(h0) = W_p @ h0 + b_p.

    Only the local predictive energy term is reported here (the write matrix
    ``B_l`` and the task energy live in the downstream nodes). Inference
    updates ``state.z_latent`` (i.e. ``z_l``) against that energy, giving the
    PC latent dynamics of the research plan.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        sigma2: float = 1.0,
        use_bias: bool = True,
        activation=None,
        energy=None,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation or IdentityActivation(),
            energy=energy or GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=MuPCInitializer(),
            sigma2=sigma2,
            use_bias=use_bias,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_weight_fan_in(source_shape: Tuple[int, ...], config: Dict) -> int:
        # f_phi is a per-position linear read: fan-in is the feature dim.
        return source_shape[-1]

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config=None):
        config = config or {}
        weight_init = weight_init or MuPCInitializer()
        use_bias = config.get("use_bias", True)
        r = node_shape[-1]
        weights = {}
        from fabricpc.core.initializers import initialize

        for edge_key, in_shape in input_shapes.items():
            w_key, key = jax.random.split(key)
            weights[edge_key] = initialize(w_key, (in_shape[-1], r), weight_init)
        biases = {"b": jnp.zeros((r,))} if use_bias else {}
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(params, inputs, state, node_info) -> NodeState:
        pre = jnp.zeros((state.z_latent.shape[0],) + node_info.shape)
        for edge_key, x in inputs.items():
            pre = pre + jnp.matmul(x, params.weights[edge_key])
        if "b" in params.biases and params.biases["b"].size > 0:
            pre = pre + params.biases["b"]
        activation = node_info.activation
        z_mu = type(activation).forward(pre, activation.config)
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return state


@dataclass(frozen=True)
class PCResidualConfig:
    """Hyper-parameters of the continuous residual fabric.

    Attributes:
        seq_len: Max sequence length (fixed per batch).
        hidden_size: Base transformer hidden dimension ``d``.
        residual_rank: Rank ``r`` of the low-rank write ``B_l : (r, d)``.
        layer_indices: Base layers the fabric attaches to.
        num_classes: Task head output size (``-1`` disables the task head).
        infer_steps: Number of PC inference iterations ``K`` (<=4 recommended).
        eta_infer: PC inference rate for the latent ``z_l``.
        sigma2: Perceptual noise variance in the local energy term.
        use_bias: Whether the predictor ``f_phi`` learns a bias.
    """

    seq_len: int = 16
    hidden_size: int = 64
    residual_rank: int = 4
    layer_indices: Tuple[int, ...] = (0,)
    num_classes: int = -1
    infer_steps: int = 4
    eta_infer: float = 0.1
    sigma2: float = 1.0
    use_bias: bool = True


def build_pc_fabric(cfg: PCResidualConfig):
    """Build the PC residual graph.

    Returns:
        A tuple ``(structure, key_map)`` where ``key_map`` maps activation-cache
        keys (``f"h{layer}"``) and ``"y"`` to graph node names, used to assemble
        the batch dict passed to the FabricPC training step.
    """
    nodes = []
    edges = []
    task_map: Dict[str, str] = {}
    key_map: Dict[str, str] = {}

    last_out = None
    for layer in cfg.layer_indices:
        h0 = IdentityNode(shape=(cfg.seq_len, cfg.hidden_size), name=f"h0_{layer}")
        z = PCResidualNode(
            shape=(cfg.seq_len, cfg.residual_rank),
            name=f"z_{layer}",
            sigma2=cfg.sigma2,
            use_bias=cfg.use_bias,
        )
        out = LinearResidual(
            shape=(cfg.seq_len, cfg.hidden_size),
            name=f"out_{layer}",
            weight_init=MuPCInitializer(),
        )
        nodes += [h0, z, out]
        edges += [
            Edge(source=h0, target=z.slot("in")),
            Edge(source=z, target=out.slot("in")),
            Edge(source=h0, target=out.slot("skip")),
        ]
        task_map[f"h0_{layer}"] = h0.name
        key_map[f"h{layer}"] = h0.name
        last_out = out

    if cfg.num_classes > 0:
        pooled = AvgPool(shape=(cfg.hidden_size,), name="pooled", global_pool=True)
        task_node = Linear(
            shape=(cfg.num_classes,),
            name="task",
            flatten_input=False,
            weight_init=MuPCInitializer(),
        )
        nodes += [pooled, task_node]
        edges += [
            Edge(source=last_out, target=pooled.slot("in")),
            Edge(source=pooled, target=task_node.slot("in")),
        ]
        task_map["y"] = task_node.name
        key_map["y"] = task_node.name

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(**task_map),
        inference=InferenceSGD(eta_infer=cfg.eta_infer, infer_steps=cfg.infer_steps),
        graph_state_initializer=FeedforwardStateInit(),
        scaling=MuPCConfig(include_output=True),
    )
    return structure, key_map