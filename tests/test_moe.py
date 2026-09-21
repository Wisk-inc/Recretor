"""Keeping a MoE donor sparse, and holding its frozen experts at four bits."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from recreator.donor import load_donor_state, resolve_donor
from recreator.graft import graft
from recreator.moe import MoESpec, SparseMLP, count_sparse_parameters
from recreator.quant import dequantize_4bit, quantization_error, quantize_4bit
from recreator.techniques.expert_merge import collect_experts

MOE = dict(
    architectures=["GptOssForCausalLM"], model_type="gpt_oss", vocab_size=256, hidden_size=64,
    intermediate_size=32, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
    head_dim=8, num_local_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
    rope_theta=150000.0, rms_norm_eps=1e-5, tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
)


def _build_moe_donor(directory, fused: bool):
    config = dict(MOE)
    hidden, inter = config["hidden_size"], config["intermediate_size"]
    heads, kv, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    experts, vocab = config["num_local_experts"], config["vocab_size"]
    generator = torch.Generator().manual_seed(1)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    state = {"model.embed_tokens.weight": rand(vocab, hidden), "model.norm.weight": torch.ones(hidden),
             "lm_head.weight": rand(vocab, hidden)}
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        state.update({
            prefix + "self_attn.q_proj.weight": rand(heads * head_dim, hidden),
            prefix + "self_attn.k_proj.weight": rand(kv * head_dim, hidden),
            prefix + "self_attn.v_proj.weight": rand(kv * head_dim, hidden),
            prefix + "self_attn.o_proj.weight": rand(hidden, heads * head_dim),
            prefix + "input_layernorm.weight": torch.ones(hidden),
            prefix + "post_attention_layernorm.weight": torch.ones(hidden),
            prefix + "mlp.router.weight": rand(experts, hidden),
        })
        if fused:
            fused_gate_up = torch.empty(experts, hidden, 2 * inter)
            for expert in range(experts):
                fused_gate_up[expert, :, :inter] = rand(inter, hidden).T
                fused_gate_up[expert, :, inter:] = rand(inter, hidden).T
            state[prefix + "mlp.experts.gate_up_proj"] = fused_gate_up
            state[prefix + "mlp.experts.down_proj"] = torch.stack([rand(hidden, inter).T for _ in range(experts)])
        else:
            for expert in range(experts):
                base = prefix + f"mlp.experts.{expert}."
                state[base + "gate_proj.weight"] = rand(inter, hidden)
                state[base + "up_proj.weight"] = rand(inter, hidden)
                state[base + "down_proj.weight"] = rand(hidden, inter)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    save_file(state, str(directory / "model.safetensors"))
    return directory


@pytest.fixture(scope="module", params=["fused", "per_expert"])
def moe_donor(tmp_path_factory, request):
    return _build_moe_donor(tmp_path_factory.mktemp(f"moe_{request.param}"), request.param == "fused")


def test_spec_reads_the_donors_sparse_shape():
    spec = MoESpec.from_donor(MOE)
    assert spec.num_experts == 8 and spec.top_k == 2 and spec.intermediate_size == 32
    assert spec.active_fraction() == 0.25
    assert MoESpec.from_donor({"hidden_size": 64}) is None


def test_sparse_graft_keeps_every_expert(moe_donor):
    result = graft(str(moe_donor), preset="faithful", mlp_mode="sparse")
    assert len(result.sparse_layers) == 2
    for layer in result.model.model.layers:
        assert isinstance(layer.mlp, SparseMLP)
        assert layer.mlp.num_experts == 8


def test_sparse_mlp_reproduces_the_donors_routed_mixture(moe_donor):
    """The experts and router are copied, so the student's MLP must compute what the donor's did."""
    result = graft(str(moe_donor), preset="faithful", mlp_mode="sparse")
    state = load_donor_state(resolve_donor(str(moe_donor)))
    stack = collect_experts(state, 0)
    router = state["model.layers.0.mlp.router.weight"]

    x = torch.randn(5, 64, generator=torch.Generator().manual_seed(3))
    weights, indices = F.linear(x, router).float().softmax(-1).topk(2, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    expected = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(2):
            expert = indices[token, slot]
            hidden = F.silu(F.linear(x[token], stack.gate[expert])) * F.linear(x[token], stack.up[expert])
            expected[token] += weights[token, slot] * F.linear(hidden, stack.down[expert])

    got = result.model.model.layers[0].mlp(x.unsqueeze(0)).squeeze(0)
    assert torch.allclose(got, expected, atol=1e-5)


def test_sparse_model_runs_forward(moe_donor):
    result = graft(str(moe_donor), preset="faithful", mlp_mode="sparse")
    logits = result.model(torch.randint(0, 256, (2, 16))).logits
    assert logits.shape == (2, 16, 256)
    assert torch.isfinite(logits).all()


def test_dense_mode_still_collapses(moe_donor):
    """Dense collapse remains available, and produces a far smaller student."""
    sparse = graft(str(moe_donor), preset="faithful", mlp_mode="sparse")
    dense = graft(str(moe_donor), preset="faithful", mlp_mode="dense")
    sparse_count = sum(p.numel() for p in sparse.model.parameters())
    dense_count = sum(p.numel() for p in dense.model.parameters())
    assert dense_count < sparse_count


def test_counting_separates_total_from_active():
    spec = MoESpec.from_donor(MOE)
    total, active = count_sparse_parameters(spec, hidden_size=64, num_layers=2)
    assert active < total
    assert active / total == pytest.approx(spec.top_k / spec.num_experts, rel=0.05)


def test_quantizing_experts_shrinks_them_and_keeps_the_model_running(moe_donor):
    result = graft(str(moe_donor), preset="faithful", mlp_mode="sparse")
    mlp = result.model.model.layers[0].mlp
    x = torch.randn(1, 8, 64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        before = mlp(x)

    held = mlp.quantize_experts()
    assert mlp.quantized
    assert held / (8 * 3 * 32 * 64) < 0.6  # under 0.6 bytes per parameter

    with torch.no_grad():
        after = mlp(x)
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_four_bit_roundtrip_is_close_and_small():
    torch.manual_seed(0)
    tensor = torch.randn(256, 256) * 0.02
    error = quantization_error(tensor)
    assert error["cosine"] > 0.99
    assert error["bytes_per_parameter"] < 0.6


def test_four_bit_survives_outliers():
    """Blockwise scaling means one huge value cannot flatten the rest of the tensor."""
    torch.manual_seed(0)
    tensor = torch.randn(128, 128) * 0.02
    tensor[0, 0] = 100.0
    error = quantization_error(tensor)
    assert error["cosine"] > 0.99


def test_four_bit_preserves_shape_and_zeros():
    tensor = torch.zeros(7, 13)
    restored = dequantize_4bit(quantize_4bit(tensor))
    assert restored.shape == tensor.shape
    assert torch.count_nonzero(restored) == 0
