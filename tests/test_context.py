"""Context scaling: what grows with N, what does not, and what that costs."""

from __future__ import annotations

from recreator.context import context_cost, context_table
from recreator.student import derive_student_config

GPT_OSS = dict(vocab_size=201088, hidden_size=2880, intermediate_size=2880, num_hidden_layers=36,
               num_attention_heads=64, num_key_value_heads=8, head_dim=64, num_local_experts=128,
               num_experts_per_tok=4, rope_theta=150000.0, rms_norm_eps=1e-5)


def _config():
    return derive_student_config(GPT_OSS, "balanced")


def test_hot_state_is_constant_in_context_length():
    """The whole point of the architecture: a decode step's resident state does not grow with N."""
    config = _config()
    small = context_cost(config, 10**6)
    huge = context_cost(config, 10**10)
    assert small.hot_state == huge.hot_state


def test_read_per_token_is_constant_in_context_length():
    config = _config()
    assert context_cost(config, 10**6).read_per_token == context_cost(config, 10**10).read_per_token


def test_cold_store_is_linear_in_context_length():
    config = _config()
    small = context_cost(config, 10**6)
    large = context_cost(config, 10**7)
    assert large.cold_store / small.cold_store == __import__("pytest").approx(10, rel=0.01)


def test_descent_grows_logarithmically():
    """A thousand-fold more context should cost only a few more levels of descent."""
    config = _config()
    shallow = context_cost(config, 10**6).descent_levels
    deep = context_cost(config, 10**9).descent_levels
    assert deep > shallow
    assert deep - shallow <= 5


def test_only_index_layers_retain_history():
    config = _config()
    cost = context_cost(config, 10**6)
    assert cost.index_layers + cost.local_layers == config.num_hidden_layers
    assert cost.index_layers < config.num_hidden_layers


def test_read_latency_scales_with_bandwidth():
    cost = context_cost(_config(), 10**9)
    assert cost.read_latency_ms(7.0) < cost.read_latency_ms(1.0)


def test_render_and_table_produce_output():
    config = _config()
    assert "cold store" in context_table(config).lower() or "cold" in context_table(config).lower()
    rendered = context_cost(config, 10**9).render()
    assert "constant in N" in rendered
    assert "prefill" in rendered
