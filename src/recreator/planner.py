"""Work out what actually fits before spending an hour finding out that it does not.

Every number here is an estimate with its assumptions written down, not a measurement. The model
is the usual one for mixed-precision training:

* **weights** -- every student parameter, at the training dtype;
* **gradients** -- one per *trainable* parameter;
* **optimizer state** -- AdamW keeps two moments plus an fp32 master copy, so 12 bytes per
  trainable parameter, or 6 with an 8-bit optimizer;
* **teacher** -- the frozen donor, at roughly 0.55 bytes per parameter once NF4 quantisation and
  its own constants are counted;
* **activations** -- what backprop has to keep alive between the forward and backward passes.

The reason a recreation fits where ordinary fine-tuning does not is the trainable fraction. A graft
freezes everything the donor supplied and trains only the strands it could not, so the three terms
that scale with *trainable* parameters -- gradients, moments, master weights -- shrink with it,
and those are what usually decide whether a run starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: Bytes per parameter for a frozen NF4 teacher: 4 bits of weight plus double-quantised constants.
NF4_BYTES_PER_PARAM = 0.55
#: AdamW: fp32 exp_avg + fp32 exp_avg_sq + fp32 master weight.
ADAMW_BYTES_PER_TRAINABLE = 12
#: bitsandbytes 8-bit moments, still with an fp32 master weight.
ADAMW_8BIT_BYTES_PER_TRAINABLE = 6
_GIB = 1024**3


def count_parameters(config) -> int:
    """Exact parameter count for a HELIX config, allocating nothing.

    The model is built on the meta device, so a 120B-parameter config costs no memory to size.
    """
    from helix_lm import HelixForCausalLM

    with torch.device("meta"):
        model = HelixForCausalLM(config)
    return sum(p.numel() for p in model.parameters())


def count_new_strand_parameters(config) -> int:
    """Parameters belonging to strands R and I -- what a frozen-trunk recreation actually trains."""
    from helix_lm import HelixForCausalLM

    from .techniques.gates import strand_parameter_names

    with torch.device("meta"):
        model = HelixForCausalLM(config)
    names = set(strand_parameter_names(model))
    return sum(p.numel() for name, p in model.named_parameters() if name in names)


def _count(value: int) -> str:
    """Format a parameter count in whichever unit keeps it readable."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= scale:
            return f"{value / scale:.2f}{suffix}"
    return str(value)


@dataclass
class MemoryPlan:
    """A per-component memory estimate, in bytes."""

    student_parameters: int
    trainable_parameters: int
    teacher_parameters: int
    weights: int
    gradients: int
    optimizer: int
    teacher: int
    activations: int
    budget: int
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.weights + self.gradients + self.optimizer + self.teacher + self.activations

    @property
    def fits(self) -> bool:
        return self.total <= self.budget

    @property
    def headroom(self) -> int:
        return self.budget - self.total

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameters / self.student_parameters if self.student_parameters else 0.0

    def render(self) -> str:
        """A human-readable breakdown."""
        rows = [
            ("student weights", self.weights),
            ("gradients", self.gradients),
            ("optimizer state", self.optimizer),
            ("teacher (frozen)", self.teacher),
            ("activations", self.activations),
        ]
        width = max(len(label) for label, _ in rows)
        lines = [
            f"student     {_count(self.student_parameters)} parameters",
            f"trainable   {_count(self.trainable_parameters)} ({self.trainable_fraction:.1%})",
            f"teacher     {_count(self.teacher_parameters)} parameters",
            "",
        ]
        lines += [f"  {label.ljust(width)}  {value / _GIB:8.2f} GiB" for label, value in rows]
        lines += [
            f"  {'TOTAL'.ljust(width)}  {self.total / _GIB:8.2f} GiB",
            f"  {'budget'.ljust(width)}  {self.budget / _GIB:8.2f} GiB",
            "",
            f"{'FITS' if self.fits else 'DOES NOT FIT'}: "
            f"{abs(self.headroom) / _GIB:.2f} GiB {'spare' if self.fits else 'over budget'}",
        ]
        lines += [f"note: {note}" for note in self.notes]
        return "\n".join(lines)


def estimate_activations(
    config,
    *,
    batch_size: int,
    seq_len: int,
    bytes_per_element: int = 2,
    gradient_checkpointing: bool = True,
) -> int:
    """Estimate activation memory held between forward and backward.

    With gradient checkpointing only each block's input is stored, plus one block's worth of
    interior activations recomputed during the backward pass. Without it, every block's interior is
    held at once. The interior multiplier is empirical -- the braid keeps the local attention
    output, the recurrent stream and the SwiGLU intermediate alive -- so treat this as the right
    order of magnitude rather than an exact byte count.
    """
    tokens = batch_size * seq_len
    stream = tokens * config.hidden_size * bytes_per_element
    interior = stream * (4 + 3 * config.intermediate_size / config.hidden_size)
    if gradient_checkpointing:
        return int(stream * config.num_hidden_layers + interior)
    return int(interior * config.num_hidden_layers)


def plan_memory(
    config,
    *,
    teacher_parameters: int,
    budget_gib: float,
    batch_size: int = 1,
    seq_len: int = 2048,
    trainable_parameters: int | None = None,
    param_bytes: int = 2,
    teacher_4bit: bool = True,
    eight_bit_optimizer: bool = False,
    gradient_checkpointing: bool = True,
    teacher_cached: bool = False,
) -> MemoryPlan:
    """Estimate whether a recreation fits the given VRAM budget.

    Args:
        config: The student's ``HelixConfig``.
        teacher_parameters: Donor parameter count.
        budget_gib: Total VRAM available, in GiB.
        batch_size / seq_len: The training shape to plan for.
        trainable_parameters: Defaults to the new strands only -- the frozen-trunk recreation.
        param_bytes: Bytes per student parameter (2 for bf16).
        teacher_4bit: Whether the teacher is NF4-quantised.
        eight_bit_optimizer: Whether to use 8-bit AdamW moments.
        gradient_checkpointing: Whether blocks are recomputed in the backward pass.
        teacher_cached: If true, the teacher's outputs were precomputed to disk and it is not
            resident during training at all.

    Returns:
        A :class:`MemoryPlan`.
    """
    student_parameters = count_parameters(config)
    if trainable_parameters is None:
        trainable_parameters = count_new_strand_parameters(config)

    notes: list[str] = []
    weights = student_parameters * param_bytes
    gradients = trainable_parameters * param_bytes
    per_trainable = ADAMW_8BIT_BYTES_PER_TRAINABLE if eight_bit_optimizer else ADAMW_BYTES_PER_TRAINABLE
    optimizer = trainable_parameters * per_trainable

    if teacher_cached:
        teacher = 0
        notes.append("teacher is not resident: its outputs are read from the on-disk cache")
    else:
        teacher = int(teacher_parameters * (NF4_BYTES_PER_PARAM if teacher_4bit else param_bytes))

    activations = estimate_activations(
        config,
        batch_size=batch_size,
        seq_len=seq_len,
        bytes_per_element=param_bytes,
        gradient_checkpointing=gradient_checkpointing,
    )
    if not gradient_checkpointing:
        notes.append("gradient checkpointing is off; turning it on is the cheapest saving available")

    plan = MemoryPlan(
        student_parameters=student_parameters,
        trainable_parameters=trainable_parameters,
        teacher_parameters=teacher_parameters,
        weights=weights,
        gradients=gradients,
        optimizer=optimizer,
        teacher=teacher,
        activations=activations,
        budget=int(budget_gib * _GIB),
        notes=notes,
    )
    if not plan.fits:
        plan.notes.extend(_suggestions(plan, eight_bit_optimizer, teacher_cached, gradient_checkpointing))
    return plan


def _suggestions(plan: MemoryPlan, eight_bit: bool, cached: bool, checkpointing: bool) -> list[str]:
    """Concrete next steps for a plan that does not fit, largest saving first."""
    options: list[tuple[int, str]] = []
    if not cached and plan.teacher:
        options.append((plan.teacher, "precompute teacher targets to disk (`--cache-teacher`) to free the teacher entirely"))
    if not eight_bit:
        saving = plan.trainable_parameters * (ADAMW_BYTES_PER_TRAINABLE - ADAMW_8BIT_BYTES_PER_TRAINABLE)
        options.append((saving, "use an 8-bit optimizer (`--eight-bit-optimizer`)"))
    if not checkpointing:
        options.append((plan.activations // 2, "enable gradient checkpointing"))
    if plan.activations > plan.total // 4:
        options.append((plan.activations // 2, "halve the sequence length or batch size, or start the context ladder lower"))
    options.append((0, "run the layer-wise schedule (`--schedule layerwise`), which holds one block at a time"))
    ranked = sorted(options, key=lambda item: -item[0])
    return [f"over by {plan.total - plan.budget:.0f} bytes; try: " + ranked[0][1]] + [
        f"  also: {text}" for _, text in ranked[1:]
    ]
