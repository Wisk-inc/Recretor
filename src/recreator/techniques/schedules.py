"""Training schedules that suit what a graft actually has to learn.

A recreation is not a pretraining run and should not be paced like one. The student begins as its
donor, and the gap between them has a specific shape: the donor's attention was global, the
student's strand L is windowed, and the strands meant to cover the difference start silent. So the
schedules here are about *where* the gap is, not about optimisation in general.

The context ladder is the important one. Strand I exists to retrieve across long context, and at
short sequence lengths there is nothing for it to retrieve -- every token is already inside strand
L's window, the index has no closed blocks to descend, and the gradient reaching its landmark
poolers is close to nothing. Training long from the start pays for that context on every step while
the student is still learning the easy part. The ladder spends early steps short and cheap, then
lengthens once the local behaviour has settled and the index has something to do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ContextLadder:
    """A sequence-length curriculum that lengthens as training proceeds.

    Lengths are rounded to whole memory blocks, because a partial block neither closes a leaf for
    the landmark tree nor fills the local staircase, and would train the index on a shape it never
    sees at inference.

    Attributes:
        start: Sequence length for the first rung.
        end: Sequence length for the last rung.
        rungs: Number of distinct lengths.
        block_size: The student's ``block_size``.
        warmup_fraction: Share of training spent on the first rung before climbing.
    """

    start: int = 512
    end: int = 32768
    rungs: int = 6
    block_size: int = 64
    warmup_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.start <= 0 or self.end < self.start:
            raise ValueError(f"Need 0 < start <= end; got start={self.start}, end={self.end}.")
        if self.rungs < 1:
            raise ValueError(f"rungs must be >= 1, got {self.rungs}.")

    def _round(self, length: float) -> int:
        blocks = max(1, round(length / self.block_size))
        return blocks * self.block_size

    @property
    def lengths(self) -> list[int]:
        """The rungs, geometrically spaced and block-aligned."""
        if self.rungs == 1:
            return [self._round(self.end)]
        ratio = (self.end / self.start) ** (1 / (self.rungs - 1))
        return [self._round(self.start * ratio**step) for step in range(self.rungs)]

    def length_at(self, fraction: float) -> int:
        """Sequence length at training progress ``fraction`` in ``[0, 1]``."""
        fraction = min(max(fraction, 0.0), 1.0)
        if fraction < self.warmup_fraction:
            return self.lengths[0]
        span = 1.0 - self.warmup_fraction
        climbed = 0.0 if span <= 0 else (fraction - self.warmup_fraction) / span
        index = min(self.rungs - 1, int(climbed * self.rungs))
        return self.lengths[index]

    def tokens_saved(self, total_steps: int, batch_size: int = 1) -> tuple[int, int]:
        """``(ladder_tokens, flat_tokens)`` -- what the ladder costs against training at ``end``.

        Useful for reporting: the ladder's saving is real but bounded, and this makes it concrete
        rather than a claim.
        """
        ladder = sum(self.length_at(step / max(1, total_steps - 1)) for step in range(total_steps))
        return ladder * batch_size, self.end * total_steps * batch_size


def cosine_with_warmup(step: int, total_steps: int, *, warmup_steps: int = 0, min_ratio: float = 0.1) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay to ``min_ratio``."""
    if total_steps <= 0:
        return 1.0
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
