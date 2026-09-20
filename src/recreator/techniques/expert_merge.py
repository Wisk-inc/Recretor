"""Collapse a mixture-of-experts feed-forward stack into the single dense MLP HELIX uses.

A HELIX block has one SwiGLU MLP. An MoE donor -- GPT-OSS, Mixtral, Qwen-MoE -- has many, and picks
a few per token with a router. There is no exact dense equivalent: the router makes the MoE layer a
piecewise function, and one matrix cannot be piecewise. What *can* be preserved is the layer's
average behaviour, and that is what merging does here.

Each expert is weighted by how often the router is expected to reach it. With no traffic statistics
the prior is uniform, which reproduces the expert mean. Given measured router probabilities --
:func:`router_prior_from_logits` computes them from real batches -- experts the model actually uses
dominate the merge and experts it has learned to ignore stop dragging the average around.

The merged MLP is an *initialisation*, not an equivalence. It is the point of the distillation
stage that follows: the merged layer starts close enough to the donor's mean behaviour for the
teacher's own outputs to pull it the rest of the way, which is far cheaper than learning a dense
feed-forward stack from noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..errors import GraftError


@dataclass
class ExpertStack:
    """One donor layer's experts, normalised to a common layout.

    Attributes:
        gate: ``(experts, intermediate, hidden)`` SwiGLU gate projections.
        up: ``(experts, intermediate, hidden)`` up projections.
        down: ``(experts, hidden, intermediate)`` down projections.
        router: ``(experts, hidden)`` router weights, when the donor exposes them.
    """

    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    router: torch.Tensor | None = None

    @property
    def num_experts(self) -> int:
        return self.gate.shape[0]


def collect_experts(state: dict[str, torch.Tensor], layer: int) -> ExpertStack | None:
    """Find layer ``layer``'s experts in a donor state dict, whatever layout the donor uses.

    Recognises the per-expert layout (Mixtral's ``experts.<i>.w1/w2/w3``) and the fused layout
    (GPT-OSS's stacked ``experts.gate_up_proj`` / ``experts.down_proj``).

    Returns:
        The stack, or ``None`` if this layer is dense.
    """
    prefix = f"model.layers.{layer}."
    keys = [key for key in state if key.startswith(prefix) and "expert" in key]
    if not keys:
        return None

    router = None
    for suffix in ("mlp.router.weight", "mlp.gate.weight", "block_sparse_moe.gate.weight"):
        if prefix + suffix in state:
            router = state[prefix + suffix]
            break

    # Fused layout: one stacked tensor per projection, experts on dim 0.
    for base in ("mlp.experts.", "block_sparse_moe.experts."):
        fused_gate_up = state.get(f"{prefix}{base}gate_up_proj")
        fused_down = state.get(f"{prefix}{base}down_proj")
        if fused_gate_up is not None and fused_down is not None:
            # `(experts, hidden, 2 * intermediate)` interleaves gate and up along the last axis.
            if fused_gate_up.shape[-1] % 2:
                raise GraftError(f"layer {layer}: fused gate_up has odd width {fused_gate_up.shape[-1]}.")
            half = fused_gate_up.shape[-1] // 2
            gate = fused_gate_up[..., :half].transpose(-1, -2).contiguous()
            up = fused_gate_up[..., half:].transpose(-1, -2).contiguous()
            down = fused_down.transpose(-1, -2).contiguous()
            return ExpertStack(gate=gate, up=up, down=down, router=router)

    # Per-expert layout: gather w1/w3/w2 (gate/up/down) across expert indices.
    indices = sorted(
        {int(key.split("experts.")[1].split(".")[0]) for key in keys if "experts." in key and key.split("experts.")[1][0].isdigit()}
    )
    if not indices:
        return None
    gates, ups, downs = [], [], []
    for index in indices:
        for base in (f"{prefix}block_sparse_moe.experts.{index}.", f"{prefix}mlp.experts.{index}."):
            triple = [
                (state.get(base + "w1"), state.get(base + "w3"), state.get(base + "w2")),
                (state.get(base + "gate_proj.weight"), state.get(base + "up_proj.weight"), state.get(base + "down_proj.weight")),
            ]
            for gate, up, down in triple:
                if gate is not None and up is not None and down is not None:
                    gates.append(gate)
                    ups.append(up)
                    downs.append(down)
                    break
            else:
                continue
            break
    if not gates:
        return None
    return ExpertStack(torch.stack(gates), torch.stack(ups), torch.stack(downs), router)


def router_prior(stack: ExpertStack, *, traffic: torch.Tensor | None = None) -> torch.Tensor:
    """Per-expert merge weights, summing to one.

    Args:
        stack: The layer's experts.
        traffic: Optional measured routing mass per expert, from
            :func:`router_prior_from_logits`. Uniform when omitted.
    """
    if traffic is None:
        return torch.full((stack.num_experts,), 1.0 / stack.num_experts, dtype=torch.float32)
    weights = traffic.detach().to(torch.float32).clamp_min(0)
    total = float(weights.sum())
    if total <= 0:
        return torch.full((stack.num_experts,), 1.0 / stack.num_experts, dtype=torch.float32)
    return weights / total


def router_prior_from_logits(router_logits: torch.Tensor, *, top_k: int | None = None) -> torch.Tensor:
    """Turn observed router logits into per-expert traffic.

    Args:
        router_logits: ``(tokens, experts)`` logits recorded from real batches.
        top_k: If set, count only the mass the router actually dispatches, which is what a
            top-k MoE layer really spends.

    Returns:
        ``(experts,)`` total routed probability mass.
    """
    probabilities = router_logits.detach().to(torch.float32).softmax(-1)
    if top_k is not None and 0 < top_k < probabilities.shape[-1]:
        values, indices = probabilities.topk(top_k, dim=-1)
        kept = torch.zeros_like(probabilities)
        kept.scatter_(-1, indices, values)
        probabilities = kept
    return probabilities.sum(0)


def merge_experts(
    stack: ExpertStack,
    *,
    traffic: torch.Tensor | None = None,
    intermediate_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Merge an expert stack into dense ``gate_proj`` / ``up_proj`` / ``down_proj`` weights.

    Args:
        stack: The layer's experts.
        traffic: Optional per-expert routing mass; uniform when omitted.
        intermediate_size: Target MLP width. When the merged width differs, the highest-traffic
            experts' rows are kept -- truncating a mean would blur every expert into the cut.

    Returns:
        A dict of dense weights ready to load into ``HelixMLP``.
    """
    weights = router_prior(stack, traffic=traffic).to(stack.gate.dtype).to(stack.gate.device)
    shaped = weights.view(-1, *([1] * (stack.gate.dim() - 1)))

    merged = {
        "gate_proj.weight": (stack.gate * shaped).sum(0),
        "up_proj.weight": (stack.up * shaped).sum(0),
        "down_proj.weight": (stack.down * shaped.view(-1, 1, 1)).sum(0),
    }

    if intermediate_size is not None and merged["gate_proj.weight"].shape[0] != intermediate_size:
        width = merged["gate_proj.weight"].shape[0]
        if intermediate_size < width:
            # Keep the rows carrying the most energy rather than the arbitrary leading block.
            energy = merged["gate_proj.weight"].to(torch.float32).norm(dim=1)
            keep = energy.topk(intermediate_size).indices.sort().values
            merged["gate_proj.weight"] = merged["gate_proj.weight"][keep]
            merged["up_proj.weight"] = merged["up_proj.weight"][keep]
            merged["down_proj.weight"] = merged["down_proj.weight"][:, keep]
        else:
            pad = intermediate_size - width
            for key in ("gate_proj.weight", "up_proj.weight"):
                tensor = merged[key]
                merged[key] = torch.cat([tensor, tensor.new_zeros(pad, tensor.shape[1])], dim=0)
            down = merged["down_proj.weight"]
            merged["down_proj.weight"] = torch.cat([down, down.new_zeros(down.shape[0], pad)], dim=1)
    return merged


def concat_experts(
    stack: ExpertStack,
    *,
    traffic: torch.Tensor | None = None,
    keep: int | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    """Widen the experts into one dense MLP that computes their weighted sum *exactly*.

    Averaging expert weights approximates a mixture by a single expert, which is wrong in a way no
    amount of training fully repairs: ``expert(mean(W))`` is not ``mean(expert(W))`` once a SwiGLU
    nonlinearity sits between them. Concatenation avoids the approximation entirely.

    A SwiGLU MLP computes ``down(silu(gate(x)) * up(x))``. Stack the kept experts' ``gate`` and
    ``up`` rows on top of each other and their ``down`` columns side by side, and the block-diagonal
    structure makes the dense layer evaluate::

        sum_i  down_i(silu(gate_i(x)) * up_i(x))

    which is the mean-field mixture -- every expert's true output, weighted by its routing mass --
    rather than a blur of their parameters. Folding the weights into ``down`` keeps the arithmetic
    in one place and leaves ``gate``/``up`` exactly as the donor trained them.

    The cost is width: keeping ``k`` experts of width ``m`` gives an MLP of width ``k * m``. That is
    the honest price of turning a sparse layer dense, and ``keep`` is where you pay it.

    Args:
        stack: The layer's experts.
        traffic: Per-expert routing mass; uniform when omitted.
        keep: How many of the highest-traffic experts to retain. ``None`` keeps them all.

    Returns:
        ``(weights, intermediate_size)`` -- dense weights and the width they imply.
    """
    weights = router_prior(stack, traffic=traffic)
    order = weights.argsort(descending=True)
    if keep is not None and 0 < keep < stack.num_experts:
        order = order[:keep]
    order = order.sort().values  # keep donor expert order for reproducibility

    chosen = weights[order].to(stack.gate.dtype).to(stack.gate.device)
    gate = torch.cat([stack.gate[i] for i in order], dim=0)
    up = torch.cat([stack.up[i] for i in order], dim=0)
    # Scale each expert's output block by its routing mass, so the concatenated sum is weighted.
    down = torch.cat([stack.down[i] * chosen[n] for n, i in enumerate(order)], dim=1)

    return (
        {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down},
        gate.shape[0],
    )
