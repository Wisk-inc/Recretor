"""Shared fixtures: a small synthetic Qwen3-shaped donor, built on disk.

The tests need a donor whose architecture is real but whose size is not, so they build one rather
than downloading anything. Every claim the package makes -- exact graft, exact expert merge,
trainable fraction -- is checkable at this scale.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

DONOR = dict(
    architectures=["Qwen3ForCausalLM"],
    model_type="qwen3",
    vocab_size=512,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=16,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
)


def _build_donor(directory, config):
    hidden = config["hidden_size"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim, inter, vocab = config["head_dim"], config["intermediate_size"], config["vocab_size"]
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    state = {
        "model.embed_tokens.weight": rand(vocab, hidden),
        "model.norm.weight": torch.ones(hidden),
        "lm_head.weight": rand(vocab, hidden),
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        state.update(
            {
                prefix + "self_attn.q_proj.weight": rand(heads * head_dim, hidden),
                prefix + "self_attn.k_proj.weight": rand(kv * head_dim, hidden),
                prefix + "self_attn.v_proj.weight": rand(kv * head_dim, hidden),
                prefix + "self_attn.o_proj.weight": rand(hidden, heads * head_dim),
                prefix + "self_attn.q_norm.weight": torch.ones(head_dim),
                prefix + "self_attn.k_norm.weight": torch.ones(head_dim),
                prefix + "mlp.gate_proj.weight": rand(inter, hidden),
                prefix + "mlp.up_proj.weight": rand(inter, hidden),
                prefix + "mlp.down_proj.weight": rand(hidden, inter),
                prefix + "input_layernorm.weight": torch.ones(hidden),
                prefix + "post_attention_layernorm.weight": torch.ones(hidden),
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    save_file(state, str(directory / "model.safetensors"))
    return directory


@pytest.fixture(scope="session")
def donor_dir(tmp_path_factory):
    """A synthetic Qwen3-shaped donor checkpoint."""
    return _build_donor(tmp_path_factory.mktemp("donor"), dict(DONOR))


@pytest.fixture(scope="session")
def donor_config():
    return dict(DONOR)


@pytest.fixture(scope="session")
def reference(donor_dir, donor_config):
    """A real transformers Qwen3 loaded from the same weights, as ground truth."""
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import load_file

    config = transformers.Qwen3Config(**{k: v for k, v in donor_config.items() if k != "architectures"})
    model = transformers.Qwen3ForCausalLM(config)
    model.load_state_dict(load_file(str(donor_dir / "model.safetensors")), strict=False)
    model.eval()
    model.requires_grad_(False)
    return model
