"""Training: the freeze, the loop, and whether distillation actually closes the gap."""

from __future__ import annotations

import torch

from recreator.distill import DistillConfig, distill, distill_layerwise, freeze_to_new_strands
from recreator.graft import graft
from recreator.student import Preset


def _narrow(preset_block: int = 32) -> Preset:
    """A deliberately narrow local window, so the donor's long-range behaviour is out of reach."""
    return Preset(block_size=preset_block, local_blocks=1, num_window_scales=1,
                  index_layer_stride=2, index_topk=4)


def test_freeze_leaves_only_the_new_strands_trainable(donor_dir):
    result = graft(str(donor_dir), preset="balanced")
    trainable, total = freeze_to_new_strands(result.model)
    assert 0 < trainable < total
    assert trainable / total < 0.4
    for name, parameter in result.model.named_parameters():
        if parameter.requires_grad:
            assert any(token in name for token in ("recurrent", "route", "pooler", "strand_gate",
                                                   "level_bias", "distance_bias")), name


def test_freeze_can_keep_extra_parameters_trainable(donor_dir):
    result = graft(str(donor_dir), preset="balanced")
    base, _ = freeze_to_new_strands(result.model)
    widened, _ = freeze_to_new_strands(result.model, also_train=("norm",))
    assert widened > base


def test_distillation_closes_the_gap_the_window_opened(donor_dir, reference):
    """The end-to-end claim: training only the new strands recovers the donor's long-range behaviour."""
    result = graft(str(donor_dir), preset=_narrow())
    freeze_to_new_strands(result.model)

    torch.manual_seed(0)
    batches = [torch.randint(0, 512, (2, 256)) for _ in range(4)]
    config = DistillConfig(steps=60, device="cpu", learning_rate=3e-3, warmup_steps=5, alpha=1.0)
    record = distill(result.model, batches, teacher=reference, config=config)

    start = sum(record.distillation_losses[:5]) / 5
    end = sum(record.distillation_losses[-5:]) / 5
    assert end < start * 0.6, f"KL only moved {start:.3f} -> {end:.3f}"


def test_distillation_requires_a_teacher_or_a_cache(donor_dir):
    result = graft(str(donor_dir), preset="balanced")
    freeze_to_new_strands(result.model)
    try:
        distill(result.model, [], config=DistillConfig(steps=1, device="cpu"))
    except ValueError as error:
        assert "teacher" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError")


def test_distillation_from_a_cache_needs_no_teacher(donor_dir, reference, tmp_path):
    from recreator.techniques.teacher_cache import TeacherCache

    result = graft(str(donor_dir), preset=_narrow())
    freeze_to_new_strands(result.model)
    batches = [torch.randint(0, 512, (1, 128)) for _ in range(2)]
    cache = TeacherCache(tmp_path / "cache")
    cache.build(reference, batches, top_k=16, device="cpu", teacher_name="reference")

    record = distill(result.model, None, cache=cache,
                     config=DistillConfig(steps=4, device="cpu", learning_rate=1e-3, warmup_steps=1))
    assert record.steps == 4
    assert all(loss == loss for loss in record.losses)  # no NaNs


def test_layerwise_training_runs_per_block(donor_dir, reference):
    result = graft(str(donor_dir), preset=_narrow())
    batches = [torch.randint(0, 512, (1, 128)) for _ in range(2)]
    records = distill_layerwise(
        result.model, reference, batches,
        config=DistillConfig(steps=5, device="cpu", learning_rate=1e-3, warmup_steps=1),
        layers=[0, 1],
    )
    assert set(records) == {0, 1}
    for record in records.values():
        assert record.steps == 5
        assert record.losses[-1] <= record.losses[0] * 2  # not diverging


def test_context_ladder_shortens_early_batches(donor_dir, reference):
    from recreator.techniques.schedules import ContextLadder

    result = graft(str(donor_dir), preset="balanced")
    freeze_to_new_strands(result.model)
    batches = [torch.randint(0, 512, (1, 256)) for _ in range(2)]
    config = DistillConfig(steps=10, device="cpu", learning_rate=1e-4, warmup_steps=1,
                           ladder=ContextLadder(start=64, end=256, rungs=2, block_size=64))
    record = distill(result.model, batches, teacher=reference, config=config)
    assert min(record.sequence_lengths) < max(record.sequence_lengths)
