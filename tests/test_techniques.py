"""The individual mechanisms, each tested against the claim it makes."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from recreator.graft import graft
from recreator.techniques.expert_merge import ExpertStack, concat_experts, merge_experts, router_prior_from_logits
from recreator.techniques.gates import BiasedStrandGate, anneal_gate_bias, gate_report
from recreator.techniques.schedules import ContextLadder, cosine_with_warmup
from recreator.techniques.teacher_cache import TeacherCache, sparse_kl_loss


def test_biased_gate_starts_almost_entirely_on_strand_l():
    gate = BiasedStrandGate(16, 8, gate_bias=4.5)
    assert torch.all(gate.index_share < 1e-3)
    # It must stay a real parameter, or the index could never open.
    assert gate.bias.requires_grad


def test_silenced_strands_contribute_nothing(donor_dir):
    result = graft(str(donor_dir), preset="faithful")
    for name, module in result.model.named_modules():
        if type(module).__name__ == "HelixRecurrentStrand":
            assert torch.count_nonzero(module.out_proj.weight) == 0, name
    assert all(share < 1e-3 for share in gate_report(result.model).values())


def test_gate_bias_anneals_towards_an_open_index(donor_dir):
    result = graft(str(donor_dir), preset="balanced")
    before = max(gate_report(result.model).values())
    anneal_gate_bias(result.model, 1.0)
    after = max(gate_report(result.model).values())
    assert after > before
    assert after == pytest.approx(0.5, abs=1e-6)


def _expert_output(stack, index, x):
    return F.linear(F.silu(F.linear(x, stack.gate[index])) * F.linear(x, stack.up[index]), stack.down[index])


def test_concat_merge_reproduces_the_mixture_exactly():
    """Concatenation is exact where weight-averaging is not; that is the whole reason it exists."""
    torch.manual_seed(0)
    experts, hidden, inter = 6, 32, 64
    stack = ExpertStack(torch.randn(experts, inter, hidden), torch.randn(experts, inter, hidden),
                        torch.randn(experts, hidden, inter))
    traffic = torch.rand(experts)
    weights = traffic / traffic.sum()
    x = torch.randn(4, hidden)
    expected = sum(weights[i] * _expert_output(stack, i, x) for i in range(experts))

    dense, width = concat_experts(stack, traffic=traffic)
    got = F.linear(F.silu(F.linear(x, dense["gate_proj.weight"])) * F.linear(x, dense["up_proj.weight"]),
                   dense["down_proj.weight"])
    assert width == experts * inter
    assert torch.allclose(got, expected, atol=1e-3)

    averaged = merge_experts(stack, traffic=traffic)
    blurred = F.linear(F.silu(F.linear(x, averaged["gate_proj.weight"])) * F.linear(x, averaged["up_proj.weight"]),
                       averaged["down_proj.weight"])
    # Averaging expert weights is not a mixture of experts, and should be visibly worse.
    assert (blurred - expected).abs().max() > (got - expected).abs().max() * 100


def test_concat_keeps_the_busiest_experts():
    torch.manual_seed(0)
    stack = ExpertStack(torch.randn(4, 8, 6), torch.randn(4, 8, 6), torch.randn(4, 6, 8))
    traffic = torch.tensor([0.1, 5.0, 0.1, 4.0])
    _, width = concat_experts(stack, traffic=traffic, keep=2)
    assert width == 2 * 8


def test_router_prior_counts_only_dispatched_mass():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    traffic = router_prior_from_logits(logits, top_k=1)
    assert traffic[2] == 0 and traffic[3] == 0
    assert traffic[0] > 0.9 and traffic[1] > 0.9


def test_context_ladder_climbs_and_aligns_to_blocks():
    ladder = ContextLadder(start=512, end=8192, rungs=5, block_size=64, warmup_fraction=0.1)
    assert ladder.length_at(0.0) == 512
    assert ladder.length_at(1.0) == 8192
    lengths = [ladder.length_at(f / 20) for f in range(21)]
    assert lengths == sorted(lengths)
    assert all(length % 64 == 0 for length in ladder.lengths)


def test_context_ladder_saves_tokens():
    ladder = ContextLadder(start=512, end=8192, rungs=5, block_size=64)
    climbed, flat = ladder.tokens_saved(total_steps=100)
    assert climbed < flat


def test_cosine_schedule_warms_up_then_decays():
    assert cosine_with_warmup(0, 100, warmup_steps=10) < 0.2
    assert cosine_with_warmup(10, 100, warmup_steps=10) == pytest.approx(1.0, abs=1e-6)
    assert cosine_with_warmup(99, 100, warmup_steps=10) < 0.2


def test_sparse_kl_is_zero_when_the_student_matches_the_teacher():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 50)
    values, indices = logits.topk(8, dim=-1)
    assert float(sparse_kl_loss(logits, values, indices)) == pytest.approx(0.0, abs=1e-6)


def test_sparse_kl_is_positive_when_they_differ():
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 50)
    values, indices = teacher.topk(8, dim=-1)
    assert float(sparse_kl_loss(torch.randn(2, 5, 50), values, indices)) > 0


def test_teacher_cache_round_trips(tmp_path):
    class Stub(torch.nn.Module):
        def forward(self, ids):
            torch.manual_seed(int(ids.sum()))
            return type("Out", (), {"logits": torch.randn(*ids.shape, 40)})()

    cache = TeacherCache(tmp_path / "cache")
    batches = [torch.randint(0, 40, (2, 16)) for _ in range(3)]
    meta = cache.build(Stub(), batches, top_k=8, device="cpu", teacher_name="stub")
    assert meta.entries == 3 and meta.top_k == 8
    assert len(cache) == 3
    entry = cache.load(0)
    assert entry["values"].shape == (2, 16, 8)
    assert cache.metadata.teacher == "stub"
