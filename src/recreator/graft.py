"""The graft: build a HELIX student and move a donor's knowledge into it.

The sequence is fixed, and each step exists because the one before it leaves something broken:

1. Derive a student config that mirrors the donor's shape (:mod:`recreator.student`).
2. Copy every tensor with a counterpart -- attention, MLP, norms, embeddings, head.
3. Merge a MoE donor's experts into the dense MLP (:mod:`recreator.techniques.expert_merge`).
4. Silence both new strands so the student's step-zero forward pass is the donor's
   (:mod:`recreator.techniques.gates`).

Step 4 is what separates this from a re-initialisation. Without it, a student that has just
received a perfect copy of its donor's attention and MLP still produces noise, because strands R
and I are writing random vectors into the residual stream those weights were tuned against. With
it, the student starts at the donor's loss and training is repair rather than learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .donor import DonorHandle, load_donor_state, resolve_donor
from .errors import GraftError
from .mapping import GLOBAL_NAMES, FitReport, GraftReport, donor_to_student, fit_tensor
from .student import derive_student_config
from .techniques.expert_merge import collect_experts, concat_experts, merge_experts
from .moe import MoESpec, install_sparse_mlps, load_sparse_experts
from .techniques.gates import DEFAULT_GATE_BIAS, install_strand_gates, silence_recurrent_strands


@dataclass
class GraftResult:
    """A grafted student and the record of how it was built."""

    model: Any
    config: Any
    report: GraftReport
    donor_id: str
    silenced_strands: list[str]
    gated_layers: list[str]
    sparse_layers: list[str] = field(default_factory=list)

    def trainable_fraction(self) -> float:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return trainable / total if total else 0.0


def _head_geometry(donor_config: dict, student_config) -> dict[str, tuple[int, int, int]]:
    """Per-projection ``(donor_heads, student_heads, head_dim)`` for grouped-query regrouping."""
    head_dim = student_config.head_dim
    donor_q = int(donor_config.get("num_attention_heads", student_config.num_attention_heads))
    donor_kv = int(donor_config.get("num_key_value_heads", donor_q))
    return {
        "q_proj": (donor_q, student_config.num_attention_heads, head_dim),
        "k_proj": (donor_kv, student_config.num_key_value_heads, head_dim),
        "v_proj": (donor_kv, student_config.num_key_value_heads, head_dim),
    }


def graft(
    donor: str | DonorHandle,
    *,
    preset: str = "balanced",
    overrides: dict[str, Any] | None = None,
    max_position_embeddings: int = 1 << 20,
    gate_bias: float = DEFAULT_GATE_BIAS,
    identity_start: bool = True,
    dtype: torch.dtype = torch.float32,
    expert_traffic: dict[int, torch.Tensor] | None = None,
    merge_mode: str = "concat",
    moe_keep_experts: int | None = None,
    mlp_mode: str = "dense",
    allow_svd: bool = True,
    device: str = "cpu",
) -> GraftResult:
    """Recreate a donor transformer as a HELIX model.

    Args:
        donor: Hub id, local checkpoint path, or an already-resolved :class:`DonorHandle`.
        preset: HELIX hyper-parameter preset -- ``"faithful"``, ``"balanced"`` or ``"long_context"``.
        overrides: HELIX config fields to force.
        max_position_embeddings: Context the student should expect.
        gate_bias: How firmly strand I is silenced at step zero.
        identity_start: Silence the new strands so the student begins as its donor. Turning this
            off is only useful for measuring what it buys.
        dtype: Student parameter dtype.
        expert_traffic: Optional per-layer router traffic for MoE donors, ``{layer: (experts,)}``.
        merge_mode: How an MoE donor's experts become one dense MLP. ``"concat"`` widens the MLP
            so it reproduces the routed mixture exactly; ``"average"`` blends the expert weights
            into the donor's own MLP width, which is far smaller and far less faithful.
        moe_keep_experts: Experts to keep under ``"concat"``. Defaults to the donor's
            ``num_experts_per_tok``, giving the student the same active width the MoE used.
        mlp_mode: For a MoE donor, ``"sparse"`` keeps its experts and router intact and swaps
            only the attention for HELIX's braid -- the whole donor is preserved. ``"dense"``
            collapses the experts per ``merge_mode``, which is far smaller but discards most of
            the donor's parameters. Ignored for a dense donor.
        allow_svd: Project mismatched widths through a truncated SVD rather than truncating.
        device: Where to build the student.

    Returns:
        A :class:`GraftResult`.

    Raises:
        GraftError: If no donor parameter could be matched at all.
    """
    from helix_lm import HelixForCausalLM

    handle = donor if isinstance(donor, DonorHandle) else resolve_donor(donor)
    donor_id = str(handle.path)

    overrides = dict(overrides or {})
    keep_experts = moe_keep_experts
    sparse = handle.is_moe and mlp_mode == "sparse"
    spec = MoESpec.from_donor(handle.config) if sparse else None
    if sparse and spec is None:
        raise GraftError("mlp_mode='sparse' needs a donor that declares experts.")
    if handle.is_moe and not sparse and merge_mode == "concat":
        # A concatenated MLP is `keep * per_expert_width` wide. Default to the number of experts the
        # router actually dispatches per token, so the dense student keeps the MoE's active width.
        if keep_experts is None:
            keep_experts = int(handle.config.get("num_experts_per_tok") or 4)
        per_expert = int(handle.config.get("moe_intermediate_size") or handle.config.get("intermediate_size") or 0)
        if per_expert and "intermediate_size" not in overrides:
            overrides["intermediate_size"] = keep_experts * per_expert

    config = derive_student_config(
        handle.config,
        preset,
        max_position_embeddings=max_position_embeddings,
        overrides=overrides,
    )
    model = HelixForCausalLM(config).to(device=device, dtype=dtype)
    sparse_layers: list[str] = []
    if sparse:
        # Swapping the dense MLP out before reading the state dict keeps the expert parameters out
        # of the name-matching pass entirely -- they are loaded from the donor's own layout instead.
        sparse_layers = install_sparse_mlps(model, spec, dtype=dtype)

    student_state = model.state_dict()
    wanted = {name for name in student_state}
    report = GraftReport(
        student_parameters=sum(p.numel() for p in model.parameters()),
    )

    geometry = _head_geometry(handle.config, config)
    donor_state = load_donor_state(handle)
    report.donor_parameters = sum(t.numel() for t in donor_state.values())

    updates: dict[str, torch.Tensor] = {}
    for donor_name, donor_tensor in donor_state.items():
        student_name = donor_to_student(donor_name)
        if student_name is None or student_name not in wanted:
            report.skipped.append(donor_name)
            continue
        kind = student_name.split(".")[-2]
        heads = geometry.get(kind, (None, None, None))
        fitted, fit_report = fit_tensor(
            student_name,
            donor_tensor.to(torch.float32),
            student_state[student_name],
            donor_heads=heads[0],
            student_heads=heads[1],
            head_dim=heads[2],
            allow_svd=allow_svd,
        )
        updates[student_name] = fitted.to(dtype)
        report.transferred.append(fit_report)
        report.transferred_parameters += fitted.numel()

    # An MoE donor has no `mlp.*_proj` to copy, so the feed-forward is still at its initialisation.
    if sparse:
        loaded, copied = load_sparse_experts(model, donor_state, num_layers=config.num_hidden_layers)
        report.transferred_parameters += copied
        report.transferred.append(
            FitReport(
                name=f"model.layers.*.mlp (sparse, {spec.num_experts} experts)",
                donor_shape=(loaded, spec.num_experts, spec.intermediate_size, config.hidden_size),
                student_shape=(loaded, spec.num_experts, spec.intermediate_size, config.hidden_size),
                method="sparse",
                notes=f"kept {spec.num_experts} experts and the router on {loaded} layers",
            )
        )
    elif handle.is_moe:
        for layer in range(config.num_hidden_layers):
            stack = collect_experts(donor_state, layer)
            if stack is None:
                continue
            traffic = (expert_traffic or {}).get(layer)
            if merge_mode == "concat":
                merged, _ = concat_experts(stack, traffic=traffic, keep=keep_experts)
            elif merge_mode == "average":
                merged = merge_experts(stack, traffic=traffic, intermediate_size=config.intermediate_size)
            else:
                raise GraftError(f"Unknown merge_mode {merge_mode!r}; use 'concat' or 'average'.")
            for suffix, tensor in merged.items():
                name = f"model.layers.{layer}.mlp.{suffix}"
                if name not in wanted:
                    continue
                fitted, fit_report = fit_tensor(
                    name, tensor.to(torch.float32), student_state[name], allow_svd=allow_svd
                )
                fit_report.notes = f"{merge_mode} of {stack.num_experts} experts"
                updates[name] = fitted.to(dtype)
                report.transferred.append(fit_report)
                report.transferred_parameters += fitted.numel()

    if not updates:
        raise GraftError(
            f"Nothing transferred from {donor_id}: no donor parameter matched a HELIX one. "
            f"The donor reports architecture {handle.architecture!r}, which may not be a "
            "Llama/Qwen-style decoder."
        )

    missing_before = {name for name in wanted if name not in updates}
    model.load_state_dict({**student_state, **updates}, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    silenced: list[str] = []
    gated: list[str] = []
    if identity_start:
        gated = install_strand_gates(model, gate_bias=gate_bias)
        silenced = silence_recurrent_strands(model)
        model.to(device=device, dtype=dtype)

    # Report what stayed at init, ignoring buffers HELIX recomputes rather than learns.
    report.missing = sorted(name for name in missing_before if "surprise_weight" not in name)

    del donor_state
    return GraftResult(
        model=model,
        config=config,
        report=report,
        donor_id=donor_id,
        silenced_strands=silenced,
        gated_layers=gated,
        sparse_layers=sparse_layers,
    )


def graft_coverage_by_group(report: GraftReport) -> dict[str, int]:
    """Parameters transferred, grouped by the kind of tensor, for a readable summary."""
    groups: dict[str, int] = {}
    for entry in report.transferred:
        name = entry.name
        if name in GLOBAL_NAMES:
            key = name.split(".")[-2]
        else:
            parts = name.split(".")
            key = ".".join(parts[3:-1]) if len(parts) > 4 else parts[-2]
        count = 1
        for dimension in entry.student_shape:
            count *= dimension
        groups[key] = groups.get(key, 0) + count
    return dict(sorted(groups.items(), key=lambda item: -item[1]))
