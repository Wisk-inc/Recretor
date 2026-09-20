"""Derive a HELIX student configuration from a donor transformer's configuration.

The rule followed here is that everything the donor and HELIX have in common is *copied*, never
re-chosen: width, depth, head counts, head dimension, MLP width, vocabulary, RoPE base and norm
epsilon all carry over untouched. If those drift, the grafted weights stop meaning what they meant
in the donor and the recreation has to be relearned from scratch rather than repaired.

Only the parameters that have no donor counterpart -- the block granularity, the local window
schedule, the landmark index and the recurrent strand -- are chosen here, and those are what the
presets in :data:`PRESETS` set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import DonorError

# Keys a donor config may use for the same quantity, in the order we trust them.
_ALIASES: dict[str, tuple[str, ...]] = {
    "vocab_size": ("vocab_size",),
    "hidden_size": ("hidden_size", "n_embd", "d_model"),
    "intermediate_size": ("intermediate_size", "ffn_dim", "n_inner"),
    "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "num_attention_heads": ("num_attention_heads", "n_head", "num_heads"),
    "num_key_value_heads": ("num_key_value_heads", "num_kv_heads", "n_head_kv"),
    "head_dim": ("head_dim",),
    "rope_theta": ("rope_theta", "rope_base"),
    "rms_norm_eps": ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"),
    "tie_word_embeddings": ("tie_word_embeddings",),
    "bos_token_id": ("bos_token_id",),
    "eos_token_id": ("eos_token_id",),
    "pad_token_id": ("pad_token_id",),
}


@dataclass(frozen=True)
class Preset:
    """A choice of the HELIX-only hyper-parameters.

    Attributes:
        block_size: Memory block granularity.
        local_blocks: Previous blocks visible to strand L.
        num_window_scales: Head groups with geometrically increasing windows. ``1`` gives every head
            the donor-like widest window, which is what makes a graft closest to its donor.
        index_layer_stride: Strand I runs on every n-th layer.
        index_topk: Memory blocks each query block retrieves.
        recurrent_fraction: Recurrent key/value width as a fraction of ``hidden_size``. Sets how many
            new parameters strand R introduces.
    """

    block_size: int = 64
    local_blocks: int = 8
    num_window_scales: int = 1
    index_layer_stride: int = 4
    index_topk: int = 8
    recurrent_fraction: float = 0.5


PRESETS: dict[str, Preset] = {
    # Closest to the donor: one window scale, so every head keeps the same wide local view, and the
    # index is sparse. Use this when you care most about how little the graft loses up front.
    "faithful": Preset(local_blocks=16, num_window_scales=1, index_layer_stride=4, index_topk=8),
    # The default trade: multi-scale windows and a denser index, a little further from the donor at
    # step zero in exchange for more of what HELIX is actually for.
    "balanced": Preset(local_blocks=8, num_window_scales=4, index_layer_stride=3, index_topk=8),
    # Cheapest per token at long context: short exact windows, heavy retrieval.
    "long_context": Preset(local_blocks=4, num_window_scales=4, index_layer_stride=2, index_topk=16),
}


def _read(donor_config: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a donor config, accepting dicts, HF config objects and aliases."""
    raw = donor_config if isinstance(donor_config, dict) else getattr(donor_config, "__dict__", {})
    for alias in _ALIASES.get(key, (key,)):
        if alias in raw and raw[alias] is not None:
            return raw[alias]
        if not isinstance(donor_config, dict):
            value = getattr(donor_config, alias, None)
            if value is not None:
                return value
    return default


def _largest_divisor_at_most(value: int, ceiling: int) -> int:
    """Largest divisor of ``value`` that is ``<= ceiling`` (always at least 1)."""
    for candidate in range(min(value, ceiling), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def derive_student_config(
    donor_config: Any,
    preset: str | Preset = "balanced",
    *,
    max_position_embeddings: int = 1 << 20,
    overrides: dict[str, Any] | None = None,
):
    """Build the :class:`helix_lm.HelixConfig` that receives the donor's weights.

    Args:
        donor_config: The donor's config -- a ``transformers`` config object or a plain dict.
        preset: A key of :data:`PRESETS`, or a :class:`Preset`.
        max_position_embeddings: Longest context the student should expect.
        overrides: HELIX config fields to force, applied last.

    Returns:
        A ``HelixConfig`` whose donor-shared fields mirror the donor exactly.

    Raises:
        DonorError: If the donor config is missing a field with no safe default.
    """
    from helix_lm import HelixConfig

    chosen = PRESETS[preset] if isinstance(preset, str) else preset
    if isinstance(preset, str) and preset not in PRESETS:  # pragma: no cover - guarded by dict lookup
        raise DonorError(f"Unknown preset {preset!r}; choose one of {sorted(PRESETS)}.")

    required = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads")
    values = {key: _read(donor_config, key) for key in required}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise DonorError(
            f"Donor config is missing {missing}, so the student cannot be shaped to match it. "
            "Pass them through `overrides` if you know them."
        )

    hidden_size = int(values["hidden_size"])
    num_heads = int(values["num_attention_heads"])
    num_kv_heads = int(_read(donor_config, "num_key_value_heads", num_heads))
    head_dim = int(_read(donor_config, "head_dim", hidden_size // num_heads))
    intermediate = int(_read(donor_config, "intermediate_size", 4 * hidden_size))

    # Strand R is new, so its width is ours to pick. Keep the head dimension aligned with the
    # donor's so the recurrent state is a familiar shape, and take as many heads as fit the budget.
    recurrent_width = max(head_dim, int(hidden_size * chosen.recurrent_fraction))
    num_recurrent_heads = max(1, recurrent_width // head_dim)

    config = HelixConfig(
        vocab_size=int(values["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=intermediate,
        num_hidden_layers=int(values["num_hidden_layers"]),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=chosen.block_size,
        local_blocks=chosen.local_blocks,
        # A scale count that does not divide the heads would silently regroup them, so clamp it to a
        # divisor rather than letting the window schedule drift off the head layout.
        num_window_scales=_largest_divisor_at_most(num_heads, chosen.num_window_scales),
        index_layer_stride=chosen.index_layer_stride,
        index_topk=chosen.index_topk,
        num_recurrent_heads=num_recurrent_heads,
        recurrent_head_dim=head_dim,
        recurrent_value_head_dim=head_dim,
        rope_theta=float(_read(donor_config, "rope_theta", 500000.0)),
        rms_norm_eps=float(_read(donor_config, "rms_norm_eps", 1e-5)),
        tie_word_embeddings=bool(_read(donor_config, "tie_word_embeddings", False)),
        max_position_embeddings=int(max_position_embeddings),
        bos_token_id=_read(donor_config, "bos_token_id", 1),
        eos_token_id=_read(donor_config, "eos_token_id", 2),
        pad_token_id=_read(donor_config, "pad_token_id", None),
    )

    for key, value in (overrides or {}).items():
        if not hasattr(config, key):
            raise DonorError(f"{key!r} is not a HelixConfig field.")
        setattr(config, key, value)
    if overrides:
        config.__post_init__()  # re-derive layer_types and the validated invariants
    return config
