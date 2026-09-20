"""The planner: sizing a recreation before running it."""

from __future__ import annotations

from recreator.planner import count_new_strand_parameters, count_parameters, plan_memory
from recreator.student import derive_student_config

QWEN3_8B = dict(vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=1000000.0,
                rms_norm_eps=1e-6, tie_word_embeddings=False)


def test_counting_allocates_nothing_even_at_scale():
    """A large config is sized on the meta device, so planning costs no memory."""
    config = derive_student_config(QWEN3_8B, "balanced")
    total = count_parameters(config)
    assert 8e9 < total < 12e9


def test_new_strands_are_a_minority_of_the_model():
    config = derive_student_config(QWEN3_8B, "balanced")
    assert 0.05 < count_new_strand_parameters(config) / count_parameters(config) < 0.4


def test_a_qwen3_8b_recreation_fits_a_96gib_card():
    config = derive_student_config(QWEN3_8B, "balanced")
    plan = plan_memory(config, teacher_parameters=8_190_000_000, budget_gib=96, seq_len=4096)
    assert plan.fits
    assert plan.trainable_fraction < 0.4
    assert "FITS" in plan.render()


def test_a_plan_that_does_not_fit_says_what_to_change():
    config = derive_student_config(QWEN3_8B, "balanced")
    plan = plan_memory(config, teacher_parameters=8_190_000_000, budget_gib=8, seq_len=4096)
    assert not plan.fits
    assert plan.notes
    assert any("cache" in note or "8-bit" in note or "layer-wise" in note for note in plan.notes)


def test_caching_the_teacher_removes_it_from_the_budget():
    config = derive_student_config(QWEN3_8B, "balanced")
    live = plan_memory(config, teacher_parameters=8_190_000_000, budget_gib=96)
    cached = plan_memory(config, teacher_parameters=8_190_000_000, budget_gib=96, teacher_cached=True)
    assert cached.teacher == 0
    assert cached.total < live.total


def test_eight_bit_optimizer_halves_the_optimizer_state():
    config = derive_student_config(QWEN3_8B, "balanced")
    full = plan_memory(config, teacher_parameters=1, budget_gib=96)
    small = plan_memory(config, teacher_parameters=1, budget_gib=96, eight_bit_optimizer=True)
    assert small.optimizer == full.optimizer // 2


def test_moe_donor_sizes_a_dense_student():
    """A MoE donor's mass is mostly in experts, so the dense student is far smaller than the donor."""
    gpt_oss = dict(vocab_size=201088, hidden_size=2880, intermediate_size=2880, num_hidden_layers=36,
                   num_attention_heads=64, num_key_value_heads=8, head_dim=64,
                   num_local_experts=128, num_experts_per_tok=4, rope_theta=150000.0, rms_norm_eps=1e-5)
    config = derive_student_config(gpt_oss, "balanced", overrides={"intermediate_size": 4 * 2880})
    assert count_parameters(config) < 20e9
