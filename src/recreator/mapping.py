"""Donor-to-HELIX parameter name mapping and shape adaptation.

HELIX names its parameters almost exactly as the Llama/Qwen family does. A block's attention lives
under ``mixer`` instead of ``self_attn``; everything else -- ``q_proj``/``k_proj``/``v_proj``/
``o_proj``, the per-head ``q_norm``/``k_norm`` that Qwen3 introduced, the SwiGLU triple
``gate_proj``/``up_proj``/``down_proj``, both block norms, ``embed_tokens``, the final ``norm`` and
``lm_head`` -- is named identically. That is why a dense Qwen3 or Llama donor transfers one tensor
at a time with no reshaping at all.

Donors that disagree on a *shape* rather than a name are handled by :func:`fit_tensor`, which
adapts rather than refuses: vocabularies are copied over their overlap, grouped-query head counts
are regrouped by averaging or repetition, and a width change is projected through a truncated SVD
so the copy keeps the donor's dominant directions instead of its arbitrary leading rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .errors import GraftError

#: Suffixes that transfer unchanged once the block prefix is rewritten.
BLOCK_SUFFIXES: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)

#: Parameters outside the block stack.
GLOBAL_NAMES: tuple[str, ...] = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def donor_to_student(name: str) -> str | None:
    """Translate a donor parameter name into its HELIX counterpart.

    Returns ``None`` when the donor parameter has no HELIX counterpart (a bias HELIX does not carry,
    a rotary buffer it recomputes, an MoE router handled separately).
    """
    if name in GLOBAL_NAMES:
        return name
    parts = name.split(".")
    if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
        suffix = ".".join(parts[3:])
        if suffix in BLOCK_SUFFIXES:
            # `self_attn` is HELIX's `mixer`; everything else keeps its name.
            return f"model.layers.{parts[2]}." + suffix.replace("self_attn.", "mixer.", 1)
    return None


@dataclass
class FitReport:
    """What :func:`fit_tensor` had to do to make a donor tensor fit."""

    name: str
    donor_shape: tuple[int, ...]
    student_shape: tuple[int, ...]
    #: ``"exact"``, ``"vocab"``, ``"gqa"``, ``"svd"``, ``"truncate"`` or ``"pad"``.
    method: str = "exact"
    notes: str = ""

    @property
    def adapted(self) -> bool:
        return self.method != "exact"


@dataclass
class GraftReport:
    """Summary of a whole transfer."""

    transferred: list[FitReport] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    donor_parameters: int = 0
    student_parameters: int = 0
    transferred_parameters: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of the student's parameters that came from the donor."""
        if self.student_parameters == 0:
            return 0.0
        return self.transferred_parameters / self.student_parameters

    def summary(self) -> str:
        adapted = sum(1 for report in self.transferred if report.adapted)
        return (
            f"grafted {len(self.transferred)} tensors "
            f"({self.transferred_parameters:,} of {self.student_parameters:,} student parameters, "
            f"{self.coverage:.1%}); {adapted} adapted, {len(self.missing)} left at init"
        )


def _regroup_heads(tensor: torch.Tensor, donor_heads: int, student_heads: int, head_dim: int) -> torch.Tensor:
    """Change a projection's head count by averaging groups down or repeating them up."""
    rest = tensor.shape[1:]
    grouped = tensor.view(donor_heads, head_dim, *rest)
    if student_heads < donor_heads:
        if donor_heads % student_heads:
            raise GraftError(f"Cannot merge {donor_heads} heads into {student_heads}: not divisible.")
        merged = grouped.view(student_heads, donor_heads // student_heads, head_dim, *rest).mean(dim=1)
    else:
        if student_heads % donor_heads:
            raise GraftError(f"Cannot expand {donor_heads} heads into {student_heads}: not divisible.")
        merged = grouped.repeat_interleave(student_heads // donor_heads, dim=0)
    return merged.reshape(student_heads * head_dim, *rest)


def _project(tensor: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Resize a 2-D tensor through a truncated SVD, keeping its strongest directions."""
    work = tensor.to(torch.float32)
    # `full_matrices=False` gives the thin factors, which is all a truncation needs.
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    rank = min(rows, cols, s.shape[0])
    out = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    fitted = torch.zeros(rows, cols, dtype=torch.float32)
    copy_rows, copy_cols = min(rows, out.shape[0]), min(cols, out.shape[1])
    fitted[:copy_rows, :copy_cols] = out[:copy_rows, :copy_cols]
    return fitted.to(tensor.dtype)


def fit_tensor(
    name: str,
    donor: torch.Tensor,
    student: torch.Tensor,
    *,
    donor_heads: int | None = None,
    student_heads: int | None = None,
    head_dim: int | None = None,
    allow_svd: bool = True,
) -> tuple[torch.Tensor, FitReport]:
    """Return a copy of ``donor`` shaped like ``student``, plus a record of what it took.

    Args:
        name: Student parameter name, for the report.
        donor: The donor tensor.
        student: The student tensor whose shape must be matched.
        donor_heads / student_heads / head_dim: Set for head-carrying projections so a
            grouped-query mismatch is regrouped rather than projected.
        allow_svd: If false, a width mismatch is truncated or zero-padded instead of projected.

    Raises:
        GraftError: If the tensors cannot be reconciled at all.
    """
    report = FitReport(name, tuple(donor.shape), tuple(student.shape))
    if donor.shape == student.shape:
        return donor.clone(), report

    if donor.dim() != student.dim():
        raise GraftError(f"{name}: donor is {donor.dim()}-D but student is {student.dim()}-D.")

    fitted = donor
    # Head regrouping first: it is the only reshape that respects what the rows *mean*.
    if (
        donor_heads
        and student_heads
        and head_dim
        and donor_heads != student_heads
        and fitted.shape[0] == donor_heads * head_dim
    ):
        fitted = _regroup_heads(fitted, donor_heads, student_heads, head_dim)
        report.method = "gqa"
        report.notes = f"{donor_heads} -> {student_heads} heads"
        if fitted.shape == student.shape:
            return fitted, report

    if fitted.dim() == 1:
        out = torch.zeros_like(student)
        width = min(fitted.shape[0], student.shape[0])
        out[:width] = fitted[:width]
        if report.method == "exact":
            report.method = "truncate" if fitted.shape[0] > student.shape[0] else "pad"
        return out, report

    if fitted.dim() != 2:
        raise GraftError(f"{name}: cannot adapt a {fitted.dim()}-D tensor of shape {tuple(fitted.shape)}.")

    rows, cols = student.shape
    # A vocabulary change is a row change on an embedding-shaped tensor: copy the shared prefix and
    # leave the rest at its initialisation, which is what a retokenised donor actually supports.
    vocab_like = fitted.shape[1] == cols and fitted.shape[0] != rows
    if vocab_like:
        out = student.clone()
        shared = min(fitted.shape[0], rows)
        out[:shared] = fitted[:shared].to(out.dtype)
        report.method = "vocab"
        report.notes = f"copied {shared:,} shared rows of {rows:,}"
        return out, report

    if allow_svd:
        report.method = "svd"
        report.notes = f"{tuple(fitted.shape)} -> {(rows, cols)}"
        return _project(fitted, rows, cols), report

    out = torch.zeros_like(student)
    copy_rows, copy_cols = min(rows, fitted.shape[0]), min(cols, fitted.shape[1])
    out[:copy_rows, :copy_cols] = fitted[:copy_rows, :copy_cols].to(out.dtype)
    report.method = "truncate"
    return out, report
