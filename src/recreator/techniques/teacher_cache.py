"""Precompute the teacher's answers so it need not be resident while the student trains.

A distillation step normally holds two models: the student being trained and the teacher producing
targets. The teacher is frozen, so it is pure overhead -- and for a large donor it is the single
biggest line in the memory plan.

Nothing forces the two to be resident at once. The teacher's output for a given batch does not
depend on the student, so it can be computed in one pass over the corpus, written to disk, and
replayed during training. The teacher is then gone and its memory belongs to the student.

Storing full logits is impractical -- a 150k vocabulary at 4096 tokens is gigabytes per batch -- so
the cache keeps the top ``k`` per position. That is a lossy but well-behaved approximation: the
tail of a trained model's distribution is nearly uniform noise, and the KL is dominated by the head.
:func:`sparse_kl_loss` renormalises over the kept entries so the target stays a proper distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class CacheMetadata:
    """What a cache directory records about how it was built."""

    teacher: str
    top_k: int
    seq_len: int
    entries: int
    vocab_size: int

    def to_dict(self) -> dict:
        return {
            "teacher": self.teacher,
            "top_k": self.top_k,
            "seq_len": self.seq_len,
            "entries": self.entries,
            "vocab_size": self.vocab_size,
        }


class TeacherCache:
    """A directory of precomputed top-k teacher logits.

    Each entry is one batch, stored as ``input_ids``, ``values`` and ``indices``.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.directory / "cache.json"

    @property
    def metadata(self) -> CacheMetadata | None:
        if not self._meta_path.is_file():
            return None
        return CacheMetadata(**json.loads(self._meta_path.read_text()))

    def entries(self) -> list[Path]:
        return sorted(self.directory.glob("batch_*.pt"))

    def __len__(self) -> int:
        return len(self.entries())

    @torch.no_grad()
    def build(
        self,
        teacher,
        batches,
        *,
        top_k: int = 64,
        device: str = "cuda",
        teacher_name: str = "unknown",
        progress=None,
    ) -> CacheMetadata:
        """Run the teacher once over ``batches`` and write its top-k logits to disk.

        Args:
            teacher: A frozen causal LM.
            batches: Iterable of ``(batch, seq)`` ``input_ids`` tensors.
            top_k: Entries kept per position.
            device: Where to run the teacher.
            teacher_name: Recorded in the metadata.
            progress: Optional callback receiving ``(index, path)``.

        Returns:
            The cache's :class:`CacheMetadata`.
        """
        count = 0
        seq_len = 0
        vocab_size = 0
        for index, input_ids in enumerate(batches):
            input_ids = input_ids.to(device)
            logits = teacher(input_ids).logits.float()
            vocab_size = logits.shape[-1]
            seq_len = input_ids.shape[1]
            values, indices = logits.topk(min(top_k, vocab_size), dim=-1)
            path = self.directory / f"batch_{index:06d}.pt"
            torch.save(
                {
                    "input_ids": input_ids.cpu(),
                    # bf16 holds a logit's magnitude to well under the noise floor of a softmax.
                    "values": values.to(torch.bfloat16).cpu(),
                    "indices": indices.to(torch.int32).cpu(),
                },
                path,
            )
            count += 1
            if progress is not None:
                progress(index, path)

        meta = CacheMetadata(
            teacher=teacher_name, top_k=top_k, seq_len=seq_len, entries=count, vocab_size=vocab_size
        )
        self._meta_path.write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
        return meta

    def load(self, index: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Load one cached batch."""
        entry = torch.load(self.entries()[index], map_location=device, weights_only=True)
        return {
            "input_ids": entry["input_ids"],
            "values": entry["values"].float(),
            "indices": entry["indices"].long(),
        }

    def __iter__(self):
        for index in range(len(self)):
            yield self.load(index)


def sparse_kl_loss(
    student_logits: torch.Tensor,
    target_values: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence against a top-k teacher distribution.

    Both sides are renormalised over the kept entries, so the target is a proper distribution over
    the teacher's own top ``k`` rather than a truncated one that silently sums to less than one.

    Args:
        student_logits: ``(batch, seq, vocab)``.
        target_values: ``(batch, seq, k)`` cached teacher logits.
        target_indices: ``(batch, seq, k)`` their vocabulary positions.
        temperature: Softens both distributions; gradients are rescaled by ``T^2`` as usual so the
            step size does not change with it.

    Returns:
        Scalar loss.
    """
    student_at_k = student_logits.gather(-1, target_indices) / temperature
    teacher_at_k = target_values / temperature
    student_log_probabilities = student_at_k.log_softmax(-1)
    teacher_probabilities = teacher_at_k.softmax(-1)
    loss = F.kl_div(student_log_probabilities, teacher_probabilities, reduction="batchmean")
    return loss * (temperature**2)
