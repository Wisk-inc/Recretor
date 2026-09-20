"""Training a grafted student back up to its donor.

Two schedules are offered, and they differ in what they hold in memory rather than in what they
optimise for.

**End-to-end** is the ordinary one: run the student, compare its logits against the teacher's, and
backpropagate. Gradients flow through the whole stack, which is what you want when the strands need
to coordinate across depth, and it costs activations proportional to depth.

**Layer-wise** trains one block at a time against the teacher's hidden states at the matching
depth. Because a HELIX block and a transformer block have the same interface -- same width in, same
width out -- block *i* of the student can be trained to reproduce block *i* of the teacher in
isolation, with no other block resident and no backward pass through the rest of the model. Peak
memory becomes a function of one block rather than the whole model, which is what makes a student
trainable on a card that could not hold its own gradients end to end. It cannot fix errors that
only appear when blocks compose, so the usual sequence is layer-wise first, then a short end-to-end
pass to settle the seams.

Both freeze the graft by default. Only the strands the donor could not supply are trained, which is
where :func:`freeze_to_new_strands` gets the trainable fraction down to roughly a sixth of the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .techniques.gates import gate_report, strand_parameter_names
from .techniques.schedules import ContextLadder, cosine_with_warmup
from .techniques.teacher_cache import sparse_kl_loss


def freeze_to_new_strands(model: nn.Module, *, also_train: Iterable[str] = ()) -> tuple[int, int]:
    """Freeze everything the donor supplied; leave the new strands trainable.

    Args:
        model: The grafted student.
        also_train: Extra parameter-name substrings to keep trainable -- ``("norm",)`` is a common
            choice, since the norms sit on the residual stream the new strands write into.

    Returns:
        ``(trainable, total)`` parameter counts.
    """
    trainable_names = set(strand_parameter_names(model))
    extra = tuple(also_train)
    for name, parameter in model.named_parameters():
        keep = name in trainable_names or any(token in name for token in extra)
        parameter.requires_grad_(keep)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@dataclass
class DistillConfig:
    """Settings for a distillation run.

    Attributes:
        steps: Optimizer steps.
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup before the cosine decay.
        batch_size: Sequences per step.
        temperature: Distillation temperature.
        alpha: Weight on the distillation term; ``1 - alpha`` goes to the language-modelling loss
            against the true next token.
        top_k: Teacher entries kept when distilling live (ignored when reading a cache).
        grad_clip: Gradient-norm clip, or ``None``.
        gradient_checkpointing: Recompute blocks in the backward pass.
        eight_bit_optimizer: Use bitsandbytes' 8-bit AdamW when available.
        ladder: Optional sequence-length curriculum.
        log_every: Steps between log callbacks.
        device: Training device.
    """

    steps: int = 1000
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    batch_size: int = 1
    temperature: float = 2.0
    alpha: float = 0.9
    top_k: int = 64
    grad_clip: float | None = 1.0
    gradient_checkpointing: bool = True
    eight_bit_optimizer: bool = False
    ladder: ContextLadder | None = None
    log_every: int = 10
    device: str = "cuda"


@dataclass
class TrainingRecord:
    """What a run produced."""

    steps: int = 0
    losses: list[float] = field(default_factory=list)
    distillation_losses: list[float] = field(default_factory=list)
    language_losses: list[float] = field(default_factory=list)
    sequence_lengths: list[int] = field(default_factory=list)
    final_gates: dict[str, float] = field(default_factory=dict)

    @property
    def final_loss(self) -> float | None:
        return self.losses[-1] if self.losses else None

    def mean_loss(self, last: int = 50) -> float | None:
        window = self.losses[-last:]
        return sum(window) / len(window) if window else None


def _build_optimizer(parameters, config: DistillConfig):
    if config.eight_bit_optimizer:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(parameters, lr=config.learning_rate, betas=(0.9, 0.95))
        except ImportError:
            import warnings

            warnings.warn(
                "bitsandbytes is not installed; falling back to torch AdamW, which needs twice the "
                "optimizer memory.",
                RuntimeWarning,
                stacklevel=2,
            )
    return torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(0.9, 0.95))


def distillation_step(
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    teacher_logits: torch.Tensor | None = None,
    cached_values: torch.Tensor | None = None,
    cached_indices: torch.Tensor | None = None,
    temperature: float = 2.0,
    alpha: float = 0.9,
    top_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine the distillation and language-modelling terms for one batch.

    The teacher may arrive either as live logits or as a cached top-k pair; live logits are reduced
    to their top ``k`` so both paths optimise the same objective.

    Returns:
        ``(total, distillation, language_modelling)``.
    """
    if teacher_logits is not None:
        values, indices = teacher_logits.float().topk(min(top_k, teacher_logits.shape[-1]), dim=-1)
    elif cached_values is not None and cached_indices is not None:
        values, indices = cached_values, cached_indices
    else:
        raise ValueError("Pass either `teacher_logits` or both `cached_values` and `cached_indices`.")

    distillation = sparse_kl_loss(student_logits, values, indices, temperature=temperature)
    language = F.cross_entropy(
        student_logits[:, :-1].reshape(-1, student_logits.shape[-1]).float(),
        input_ids[:, 1:].reshape(-1),
    )
    return alpha * distillation + (1 - alpha) * language, distillation, language


def distill(
    student: nn.Module,
    batches: Iterator[torch.Tensor] | Iterable[torch.Tensor],
    *,
    teacher: nn.Module | None = None,
    cache=None,
    config: DistillConfig | None = None,
    on_log: Callable[[int, dict[str, Any]], None] | None = None,
) -> TrainingRecord:
    """Train a grafted student end to end against a teacher.

    Args:
        student: The grafted HELIX model, already frozen to its new strands.
        batches: Iterable of ``(batch, seq)`` ``input_ids``. Ignored when ``cache`` is given.
        teacher: A frozen teacher producing live targets. Omit when using ``cache``.
        cache: A :class:`~recreator.techniques.teacher_cache.TeacherCache` to replay instead.
        config: Run settings.
        on_log: Callback receiving ``(step, metrics)``.

    Returns:
        A :class:`TrainingRecord`.

    Raises:
        ValueError: If neither a teacher nor a cache is supplied.
    """
    if teacher is None and cache is None:
        raise ValueError("Distillation needs either a live `teacher` or a precomputed `cache`.")

    config = config or DistillConfig()
    record = TrainingRecord()
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters; call `freeze_to_new_strands` with something to train.")
    optimizer = _build_optimizer(trainable, config)

    source = iter(batches) if batches is not None else iter(())
    cache_length = len(cache) if cache is not None else 0

    for step in range(config.steps):
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * cosine_with_warmup(
                step, config.steps, warmup_steps=config.warmup_steps
            )

        if cache is not None:
            entry = cache.load(step % cache_length, device=config.device)
            input_ids = entry["input_ids"]
            values, indices, teacher_logits = entry["values"], entry["indices"], None
        else:
            try:
                input_ids = next(source)
            except StopIteration:
                source = iter(batches)
                input_ids = next(source)
            input_ids = input_ids.to(config.device)
            if config.ladder is not None:
                length = min(input_ids.shape[1], config.ladder.length_at(step / max(1, config.steps - 1)))
                input_ids = input_ids[:, :length]
            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits
            values = indices = None

        student_logits = student(input_ids).logits
        total, distillation, language = distillation_step(
            student_logits,
            input_ids,
            teacher_logits=teacher_logits,
            cached_values=values,
            cached_indices=indices,
            temperature=config.temperature,
            alpha=config.alpha,
            top_k=config.top_k,
        )

        total.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        record.steps += 1
        record.losses.append(float(total.detach()))
        record.distillation_losses.append(float(distillation.detach()))
        record.language_losses.append(float(language.detach()))
        record.sequence_lengths.append(int(input_ids.shape[1]))

        if on_log is not None and (step % config.log_every == 0 or step == config.steps - 1):
            on_log(
                step,
                {
                    "loss": record.losses[-1],
                    "kl": record.distillation_losses[-1],
                    "lm": record.language_losses[-1],
                    "seq_len": record.sequence_lengths[-1],
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )

    record.final_gates = gate_report(student)
    return record


def distill_layerwise(
    student: nn.Module,
    teacher: nn.Module,
    batches: Iterable[torch.Tensor],
    *,
    config: DistillConfig | None = None,
    layers: Iterable[int] | None = None,
    on_log: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[int, TrainingRecord]:
    """Train each block against the teacher's hidden state at the matching depth.

    Only one student block is unfrozen at a time and no gradient crosses a block boundary, so peak
    memory is set by the widest block rather than by the depth of the model.

    Args:
        student: The grafted student.
        teacher: A frozen teacher that can return hidden states.
        batches: Iterable of ``(batch, seq)`` ``input_ids``.
        config: Run settings; ``steps`` is per block.
        layers: Which block indices to train. Defaults to all of them.
        on_log: Callback receiving ``(step, metrics)``; metrics carry a ``layer`` key.

    Returns:
        One :class:`TrainingRecord` per trained block.
    """
    config = config or DistillConfig()
    blocks = list(student.model.layers)
    indices = list(layers) if layers is not None else list(range(len(blocks)))
    records: dict[int, TrainingRecord] = {}
    cached = list(batches)

    trainable_names = set(strand_parameter_names(student))
    for layer_index in indices:
        block = blocks[layer_index]
        prefix = f"model.layers.{layer_index}."
        for name, parameter in student.named_parameters():
            parameter.requires_grad_(name.startswith(prefix) and name in trainable_names)
        parameters = [p for p in block.parameters() if p.requires_grad]
        if not parameters:
            continue

        optimizer = _build_optimizer(parameters, config)
        record = TrainingRecord()
        rotary = student.model.rotary_emb

        for step in range(config.steps):
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * cosine_with_warmup(
                    step, config.steps, warmup_steps=config.warmup_steps
                )
            input_ids = cached[step % len(cached)].to(config.device)

            with torch.no_grad():
                hidden_states = teacher(input_ids, output_hidden_states=True).hidden_states
                # `hidden_states[i]` is this block's input and `[i + 1]` its output, so the pair
                # defines the function the student's block has to reproduce.
                block_input = hidden_states[layer_index].to(config.device)
                target = hidden_states[layer_index + 1].to(config.device)

            positions = torch.arange(block_input.shape[1], device=config.device).unsqueeze(0)
            predicted = block(block_input, position_embeddings=rotary(block_input, positions))
            loss = F.mse_loss(predicted.float(), target.float())

            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            record.steps += 1
            record.losses.append(float(loss.detach()))
            record.sequence_lengths.append(int(input_ids.shape[1]))
            if on_log is not None and (step % config.log_every == 0 or step == config.steps - 1):
                on_log(step, {"layer": layer_index, "loss": record.losses[-1]})

        records[layer_index] = record

    return records
