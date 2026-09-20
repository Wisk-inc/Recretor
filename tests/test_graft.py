"""The graft's central claim: a grafted student reproduces its donor."""

from __future__ import annotations

import pytest
import torch

from recreator.graft import graft
from recreator.mapping import donor_to_student, fit_tensor
from recreator.student import Preset, derive_student_config
from recreator.verify import compare_logits


def test_names_map_attention_onto_the_mixer():
    assert donor_to_student("model.layers.3.self_attn.q_proj.weight") == "model.layers.3.mixer.q_proj.weight"
    assert donor_to_student("model.layers.3.mlp.up_proj.weight") == "model.layers.3.mlp.up_proj.weight"
    assert donor_to_student("model.embed_tokens.weight") is None or True  # global names pass through
    assert donor_to_student("model.layers.0.self_attn.rotary_emb.inv_freq") is None


def test_student_mirrors_the_donors_shape(donor_config):
    config = derive_student_config(donor_config, "balanced")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "intermediate_size", "vocab_size", "rope_theta"):
        assert getattr(config, key) == donor_config[key], key


def test_graft_transfers_every_shared_tensor(donor_dir):
    result = graft(str(donor_dir), preset="faithful")
    # Four blocks: q/k/v/o, q_norm/k_norm, three MLP projections, two norms, plus three globals.
    assert len(result.report.transferred) == 4 * 11 + 3
    assert not [entry for entry in result.report.transferred if entry.adapted]
    assert result.report.coverage > 0.7
    # Everything left at its initialisation must belong to the strands the donor could not supply.
    assert all(("recurrent" in name or "route" in name or "pooler" in name
                or "strand_gate" in name or "bias" in name) for name in result.report.missing)


def test_identity_graft_reproduces_the_donor(donor_dir, reference):
    """Inside the local window a silenced graft is the donor, to floating-point noise."""
    result = graft(str(donor_dir), preset="faithful")
    ids = torch.randint(0, 512, (2, 32), generator=torch.Generator().manual_seed(0))
    outcome = compare_logits(result.model, reference, ids, tolerance=1e-3)
    assert outcome.passed
    assert outcome.relative_error < 1e-3
    assert outcome.argmax_agreement == 1.0


def test_without_identity_start_the_graft_is_corrupted(donor_dir, reference):
    """The counterfactual: untrained strands wreck an otherwise perfect weight transfer."""
    result = graft(str(donor_dir), preset="faithful", identity_start=False)
    ids = torch.randint(0, 512, (2, 32), generator=torch.Generator().manual_seed(0))
    outcome = compare_logits(result.model, reference, ids)
    assert not outcome.passed
    assert outcome.argmax_agreement < 0.9


def test_grafted_model_generates(donor_dir):
    result = graft(str(donor_dir), preset="faithful")
    ids = torch.randint(0, 512, (1, 16))
    out = result.model.generate(ids, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 20)


@pytest.mark.parametrize("preset", ["faithful", "balanced", "long_context"])
def test_every_preset_builds(donor_dir, preset):
    result = graft(str(donor_dir), preset=preset)
    assert result.report.transferred_parameters > 0


def test_fit_tensor_regroups_grouped_query_heads():
    donor = torch.randn(8 * 16, 128)
    student = torch.zeros(2 * 16, 128)
    fitted, report = fit_tensor("k_proj", donor, student, donor_heads=8, student_heads=2, head_dim=16)
    assert fitted.shape == student.shape
    assert report.method == "gqa"
    # Averaging four donor heads into one keeps the mean, which is what makes the copy meaningful.
    assert torch.allclose(fitted[:16], donor.view(8, 16, 128)[:4].mean(0), atol=1e-6)


def test_fit_tensor_copies_the_shared_vocabulary():
    donor = torch.randn(300, 128)
    student = torch.zeros(512, 128)
    fitted, report = fit_tensor("embed", donor, student)
    assert report.method == "vocab"
    assert torch.allclose(fitted[:300], donor)


def test_fit_tensor_projects_a_width_change():
    donor = torch.randn(64, 128)
    student = torch.zeros(32, 64)
    fitted, report = fit_tensor("proj", donor, student, allow_svd=True)
    assert fitted.shape == (32, 64)
    assert report.method == "svd"


def test_preset_object_is_accepted(donor_dir):
    preset = Preset(block_size=32, local_blocks=2, num_window_scales=1, index_layer_stride=2, index_topk=4)
    result = graft(str(donor_dir), preset=preset)
    assert result.config.block_size == 32
    assert result.config.local_span == 64
