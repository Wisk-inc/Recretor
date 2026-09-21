"""Keep a mixture-of-experts donor sparse instead of collapsing it.

HELIX changes how a block *mixes over the sequence*. Whether that block's feed-forward is one dense
MLP or a routed bank of experts is an orthogonal choice, and nothing in the braid depends on it. So
a MoE donor does not have to be flattened to be recreated: its experts and its router can be copied
across verbatim, and only the attention is replaced.

That matters because flattening is not cheap. GPT-OSS-120B holds roughly 117B parameters, of which
about 5B are active per token -- nearly all the mass is in the experts. Collapsing them into one
dense MLP produces a student of a few billion parameters and discards the rest, permanently. Keeping
them sparse keeps the whole model, and leaves the recreation with exactly the job it should have:
learning the two new strands.

:class:`SparseMLP` is shape-compatible with ``HelixMLP`` -- one tensor in, one tensor out -- so
:func:`install_sparse_mlps` can swap it into a built HELIX model without touching the block's
forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .errors import GraftError
from .quant import DEFAULT_BLOCK_SIZE, QuantizedTensor, dequantize_4bit, quantize_4bit
from .techniques.expert_merge import ExpertStack, collect_experts


@dataclass
class MoESpec:
    """The sparse feed-forward shape read off a donor's config.

    Attributes:
        num_experts: Experts per layer.
        top_k: Experts each token is routed to.
        intermediate_size: Width of one expert.
        norm_topk_prob: Whether the router's top-k weights are renormalised to sum to one.
    """

    num_experts: int
    top_k: int
    intermediate_size: int
    norm_topk_prob: bool = True

    @classmethod
    def from_donor(cls, config: dict) -> MoESpec | None:
        """Read a MoE spec from a donor config, or ``None`` if the donor is dense."""
        experts = (
            config.get("num_local_experts")
            or config.get("num_experts")
            or config.get("n_routed_experts")
        )
        if not experts:
            return None
        width = (
            config.get("moe_intermediate_size")
            or config.get("expert_intermediate_size")
            or config.get("intermediate_size")
        )
        if not width:
            raise GraftError("Donor declares experts but no expert width.")
        return cls(
            num_experts=int(experts),
            top_k=int(config.get("num_experts_per_tok") or config.get("top_k") or 4),
            intermediate_size=int(width),
            norm_topk_prob=bool(config.get("norm_topk_prob", True)),
        )

    def active_fraction(self) -> float:
        """Share of expert parameters a single token actually touches."""
        return self.top_k / self.num_experts


class SparseMLP(nn.Module):
    """A routed bank of SwiGLU experts, drop-in compatible with ``HelixMLP``.

    Experts are held as stacked parameters rather than a ``ModuleList``, which keeps the checkpoint
    layout close to the fused form the larger MoE donors ship and avoids thousands of small modules.

    Tokens are dispatched expert by expert: each expert sees only the rows routed to it, so the
    arithmetic scales with ``top_k`` rather than with ``num_experts``. That is the whole point of a
    sparse layer, and a dense implementation would quietly give it up.
    """

    def __init__(self, spec: MoESpec, hidden_size: int, *, hidden_act: str = "silu", dtype=None) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_size = hidden_size
        self.num_experts = spec.num_experts
        self.top_k = min(spec.top_k, spec.num_experts)
        self.intermediate_size = spec.intermediate_size
        self.activation = F.silu if hidden_act == "silu" else F.gelu

        factory = {"dtype": dtype} if dtype is not None else {}
        self.quantized = False
        self._quantized: dict[str, QuantizedTensor] = {}
        self.router = nn.Linear(hidden_size, spec.num_experts, bias=False, **factory)
        self.gate_proj = nn.Parameter(torch.empty(spec.num_experts, spec.intermediate_size, hidden_size, **factory))
        self.up_proj = nn.Parameter(torch.empty(spec.num_experts, spec.intermediate_size, hidden_size, **factory))
        self.down_proj = nn.Parameter(torch.empty(spec.num_experts, hidden_size, spec.intermediate_size, **factory))
        self.reset_parameters()

    def reset_parameters(self, std: float = 0.02) -> None:
        with torch.no_grad():
            for parameter in (self.gate_proj, self.up_proj, self.down_proj):
                parameter.normal_(0.0, std)
            self.router.weight.normal_(0.0, std)

    def route(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weights, indices)`` of shape ``(tokens, top_k)``."""
        logits = self.router(flat).float()
        weights, indices = logits.softmax(-1).topk(self.top_k, dim=-1)
        if self.spec.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return weights.to(flat.dtype), indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        weights, indices = self.route(flat)

        out = torch.zeros_like(flat)
        # `one_hot` over (tokens, top_k, experts) lets one nonzero() find every token an expert owns.
        assignment = F.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)
        for expert in range(self.num_experts):
            slot, token = torch.where(assignment[expert])
            if token.numel() == 0:
                continue
            rows = flat[token]
            gate = self._expert_weight("gate_proj", expert, rows.dtype)
            up = self._expert_weight("up_proj", expert, rows.dtype)
            down = self._expert_weight("down_proj", expert, rows.dtype)
            hidden_out = self.activation(F.linear(rows, gate)) * F.linear(rows, up)
            contribution = F.linear(hidden_out, down) * weights[token, slot, None]
            out.index_add_(0, token, contribution.to(out.dtype))
        return out.view(batch, seq, hidden)

    @torch.no_grad()
    def load_stack(self, stack: ExpertStack) -> None:
        """Copy a donor's experts and router into this layer."""
        if stack.num_experts != self.num_experts:
            raise GraftError(f"Donor layer has {stack.num_experts} experts, student has {self.num_experts}.")
        for name, tensor in (("gate_proj", stack.gate), ("up_proj", stack.up), ("down_proj", stack.down)):
            target = getattr(self, name)
            if tensor.shape != target.shape:
                raise GraftError(f"{name}: donor {tuple(tensor.shape)} != student {tuple(target.shape)}.")
            target.copy_(tensor.to(target.dtype))
        if stack.router is not None:
            if stack.router.shape != self.router.weight.shape:
                raise GraftError(
                    f"router: donor {tuple(stack.router.shape)} != student {tuple(self.router.weight.shape)}."
                )
            self.router.weight.copy_(stack.router.to(self.router.weight.dtype))


    @torch.no_grad()
    def quantize_experts(self, *, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
        """Store the experts at four bits, dequantizing one expert at a time in the forward pass.

        The experts are frozen -- a recreation trains the strands, never the donor's feed-forward --
        so they need to be readable, not differentiable. Holding them packed and unpacking only the
        expert currently being applied keeps one expert's worth of dense weights alive at a time
        instead of all of them, which is what lets a 117B-parameter donor sit on a single card.

        Returns:
            Bytes now held by the expert weights.
        """
        held = 0
        for name in ("gate_proj", "up_proj", "down_proj"):
            parameter = getattr(self, name)
            packed = quantize_4bit(parameter.data, block_size=block_size)
            self._quantized[name] = packed
            held += packed.nbytes()
            # Drop the dense parameter; `_expert_weight` reads the packed copy from here on.
            setattr(self, name, None)
            delattr(self, name)
            self.register_buffer(f"{name}_codes", packed.codes)
            self.register_buffer(f"{name}_absmax", packed.absmax)
        self.quantized = True
        return held

    def _expert_weight(self, name: str, expert: int, dtype: torch.dtype) -> torch.Tensor:
        """The dense weight for one expert, unpacking it first when the bank is quantized."""
        if not self.quantized:
            return getattr(self, name)[expert]
        packed = self._quantized[name]
        rows = packed.shape[1]
        columns = packed.shape[2]
        per_expert = rows * columns
        # Slice this expert's span out of the packed buffer rather than unpacking the whole bank.
        start, stop = expert * per_expert, (expert + 1) * per_expert
        block = packed.block_size
        first_block, last_block = start // block, (stop + block - 1) // block
        codes = packed.codes[first_block * block // 2 : last_block * block // 2]
        slice_view = QuantizedTensor(
            codes=codes,
            absmax=packed.absmax[first_block:last_block],
            shape=torch.Size([(last_block - first_block) * block]),
            block_size=block,
            dtype=dtype,
        )
        offset = start - first_block * block
        return dequantize_4bit(slice_view, dtype=dtype)[offset : offset + per_expert].view(rows, columns)

    def extra_repr(self) -> str:
        state = ", 4-bit" if self.quantized else ""
        return f"experts={self.num_experts}, top_k={self.top_k}, intermediate={self.intermediate_size}{state}"


def install_sparse_mlps(model: nn.Module, spec: MoESpec, *, dtype=None) -> list[str]:
    """Replace every block's dense ``mlp`` with a :class:`SparseMLP`.

    Args:
        model: A built ``HelixForCausalLM``.
        spec: The donor's sparse feed-forward shape.
        dtype: Parameter dtype; defaults to the model's.

    Returns:
        The names of the blocks that were converted.
    """
    converted: list[str] = []
    hidden_act = getattr(model.config, "hidden_act", "silu")
    for name, module in model.named_modules():
        if type(module).__name__ != "HelixDecoderLayer":
            continue
        reference = module.mlp.gate_proj.weight
        module.mlp = SparseMLP(
            spec,
            model.config.hidden_size,
            hidden_act=hidden_act,
            dtype=dtype if dtype is not None else reference.dtype,
        ).to(reference.device)
        converted.append(f"{name}.mlp")
    return converted


def load_sparse_experts(
    model: nn.Module,
    donor_state: dict[str, torch.Tensor],
    *,
    num_layers: int,
) -> tuple[int, int]:
    """Copy every layer's experts from a donor state dict into the student's sparse MLPs.

    Returns:
        ``(layers_loaded, parameters_copied)``.
    """
    layers = list(model.model.layers)
    loaded = 0
    copied = 0
    for index in range(num_layers):
        stack = collect_experts(donor_state, index)
        if stack is None:
            continue
        mlp = layers[index].mlp
        if not isinstance(mlp, SparseMLP):
            raise GraftError(f"layer {index} has no SparseMLP; call install_sparse_mlps first.")
        mlp.load_stack(stack)
        loaded += 1
        copied += stack.gate.numel() + stack.up.numel() + stack.down.numel()
        if stack.router is not None:
            copied += stack.router.numel()
    return loaded, copied


def sparse_parameter_names(model: nn.Module) -> list[str]:
    """Names of the expert and router parameters, so a freeze can leave them alone."""
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, SparseMLP):
            names.extend(f"{name}.{parameter}" for parameter, _ in module.named_parameters(recurse=True))
    return sorted(names)


def count_sparse_parameters(spec: MoESpec, hidden_size: int, num_layers: int) -> tuple[int, int]:
    """``(total, active_per_token)`` parameters the sparse feed-forward stack contributes."""
    per_expert = 3 * spec.intermediate_size * hidden_size
    total = num_layers * (spec.num_experts * per_expert + hidden_size * spec.num_experts)
    active = num_layers * (spec.top_k * per_expert + hidden_size * spec.num_experts)
    return total, active
