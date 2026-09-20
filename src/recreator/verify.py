"""Check that a graft actually reproduces its donor.

The claim a recreation rests on is falsifiable, so it should be tested rather than asserted: with
the new strands silenced, a HELIX block computes what its donor's block computed, provided every
token is inside strand L's window. Feed both models the same tokens at a length within that window
and their logits should agree to floating-point noise.

Outside the window the two genuinely differ -- the donor attends globally and strand L does not --
and that divergence is the thing distillation repairs. Measuring it is useful too, which is why
:func:`window_sweep` reports agreement as a function of length instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .errors import RecreatorError


@dataclass
class VerifyOutcome:
    """The result of comparing a graft against its donor."""

    seq_len: int
    max_abs_diff: float
    relative_error: float
    kl: float
    argmax_agreement: float
    tolerance: float
    local_span: int
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.relative_error <= self.tolerance

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"identity graft check at {self.seq_len} tokens (local span {self.local_span}):",
            f"  max abs difference  {self.max_abs_diff:.3e}",
            f"  relative error      {self.relative_error:.3e}  (tolerance {self.tolerance:.1e})",
            f"  KL(donor || graft)  {self.kl:.3e}",
            f"  argmax agreement    {self.argmax_agreement:.2%}",
            f"  {verdict}",
        ]
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)


def _load_reference(donor_id: str):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise RecreatorError("transformers is required to verify a graft; install `recreator[train]`.") from exc

    reference = AutoModelForCausalLM.from_pretrained(donor_id, dtype=torch.float32)
    reference.eval()
    reference.requires_grad_(False)
    return reference


@torch.no_grad()
def compare_logits(
    student,
    reference,
    input_ids: torch.Tensor,
    *,
    tolerance: float = 1e-3,
    local_span: int = 0,
) -> VerifyOutcome:
    """Compare a grafted student's logits against a reference model's on the same tokens."""
    student.eval()
    donor_logits = reference(input_ids).logits.float()
    student_logits = student(input_ids).logits.float()

    difference = donor_logits - student_logits
    relative = float(difference.norm() / donor_logits.norm().clamp_min(1e-12))
    # KL is non-negative in exact arithmetic; a tiny negative here is float error, not a result.
    kl = max(
        0.0,
        float(
            F.kl_div(
                student_logits.log_softmax(-1),
                donor_logits.log_softmax(-1),
                log_target=True,
                reduction="batchmean",
            )
        ),
    )
    outcome = VerifyOutcome(
        seq_len=int(input_ids.shape[1]),
        max_abs_diff=float(difference.abs().max()),
        relative_error=relative,
        kl=kl,
        argmax_agreement=float((donor_logits.argmax(-1) == student_logits.argmax(-1)).float().mean()),
        tolerance=tolerance,
        local_span=local_span,
    )
    if local_span and input_ids.shape[1] > local_span:
        outcome.notes.append(
            f"sequence is longer than the local span ({local_span}), so divergence is expected: "
            "this is the gap distillation closes, not a failed graft"
        )
    return outcome


def verify_against_donor(
    result,
    donor_id: str,
    *,
    seq_len: int = 0,
    batch_size: int = 2,
    tolerance: float = 1e-3,
    seed: int = 0,
) -> VerifyOutcome:
    """Load the donor and check the graft reproduces it inside the local window.

    Args:
        result: A :class:`~recreator.graft.GraftResult`.
        donor_id: The donor, for loading a reference implementation.
        seq_len: Comparison length. ``0`` picks the largest whole number of blocks that still fits
            the narrowest head's window, which is where the claim actually applies.
        batch_size: Sequences to compare.
        tolerance: Maximum relative error to count as a pass.
        seed: Seed for the random tokens.

    Returns:
        A :class:`VerifyOutcome`.
    """
    config = result.config
    narrowest = min(config.local_window_sizes)
    if seq_len <= 0:
        seq_len = max(config.block_size, narrowest)

    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), generator=generator)

    reference = _load_reference(donor_id)
    outcome = compare_logits(
        result.model, reference, input_ids, tolerance=tolerance, local_span=narrowest
    )
    if not result.silenced_strands:
        outcome.notes.append(
            "this graft was built with identity_start=False, so the new strands are active and "
            "divergence from the donor is expected"
        )
    return outcome


@torch.no_grad()
def window_sweep(
    result,
    reference,
    *,
    lengths: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    batch_size: int = 1,
    seed: int = 0,
) -> list[VerifyOutcome]:
    """Measure agreement with the donor as sequence length crosses the local window.

    Inside the window the graft is exact; beyond it the donor's global attention sees what strand L
    cannot, and agreement falls away. The shape of that falloff is a direct readout of how much
    work the new strands have to do.
    """
    config = result.config
    narrowest = min(config.local_window_sizes)
    generator = torch.Generator().manual_seed(seed)
    outcomes = []
    for length in lengths:
        if length > config.max_position_embeddings:
            continue
        input_ids = torch.randint(0, config.vocab_size, (batch_size, length), generator=generator)
        outcomes.append(compare_logits(result.model, reference, input_ids, local_span=narrowest))
    return outcomes
