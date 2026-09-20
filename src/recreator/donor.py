"""Loading a donor checkpoint without paying for it twice.

A donor is only ever read: its tensors are copied into the student, and -- during distillation --
its forward pass supplies targets. Neither job needs the donor in full precision or even fully
resident, so this module keeps two paths apart:

* :func:`load_donor_state` streams the checkpoint's tensors from disk one shard at a time. Nothing
  but the shard being read is in memory, so a donor far larger than RAM can still be grafted.
* :func:`load_teacher` materialises a runnable model, quantised to 4-bit when bitsandbytes is
  present, for the distillation stage that needs real outputs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .errors import DonorError


@dataclass
class DonorHandle:
    """A donor resolved to a local directory, with its config parsed."""

    path: Path
    config: dict[str, Any]
    shards: list[Path]

    @property
    def architecture(self) -> str:
        architectures = self.config.get("architectures") or []
        return architectures[0] if architectures else self.config.get("model_type", "unknown")

    @property
    def is_moe(self) -> bool:
        keys = ("num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok")
        return any(self.config.get(key) for key in keys)


def resolve_donor(model_id: str, *, revision: str | None = None, cache_dir: str | None = None) -> DonorHandle:
    """Resolve a hub id or local path to a :class:`DonorHandle`, downloading weights if needed.

    Args:
        model_id: A Hugging Face repo id, or a path to a local checkpoint directory.
        revision: Optional git revision on the hub.
        cache_dir: Where downloads land.

    Raises:
        DonorError: If the checkpoint has no config or no readable weights.
    """
    path = Path(model_id)
    if not path.is_dir():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise DonorError(
                f"{model_id!r} is not a local directory and huggingface_hub is not installed. "
                "Install `recreator[hub]` or pass a local path."
            ) from exc
        path = Path(
            snapshot_download(
                model_id,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
            )
        )

    config_path = path / "config.json"
    if not config_path.is_file():
        raise DonorError(f"No config.json under {path}.")
    config = json.loads(config_path.read_text())
    # Some repos nest the language model's config (multimodal checkpoints, mostly).
    if "text_config" in config and "hidden_size" not in config:
        config = {**config, **config["text_config"]}

    shards = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
    if not shards:
        raise DonorError(f"No .safetensors or .bin weight files under {path}.")
    return DonorHandle(path=path, config=config, shards=shards)


def iter_donor_tensors(handle: DonorHandle) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every donor parameter, one shard at a time.

    Safetensors shards are memory-mapped and read key by key, so peak memory stays at roughly one
    tensor rather than one shard.
    """
    for shard in handle.shards:
        if shard.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(shard), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    yield key, reader.get_tensor(key)
        else:
            state = torch.load(shard, map_location="cpu", weights_only=True)
            for key, tensor in state.items():
                yield key, tensor
            del state


def load_donor_state(handle: DonorHandle, *, keep: set[str] | None = None) -> dict[str, torch.Tensor]:
    """Read a donor's tensors into a dict.

    Args:
        handle: The resolved donor.
        keep: If given, only these parameter names are retained -- which is how a graft reads a
            checkpoint much larger than memory.
    """
    state: dict[str, torch.Tensor] = {}
    for name, tensor in iter_donor_tensors(handle):
        if keep is None or name in keep:
            state[name] = tensor
    return state


def load_teacher(
    model_id: str,
    *,
    load_in_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict = "auto",
    revision: str | None = None,
):
    """Load the donor as a frozen teacher for distillation.

    Quantises to NF4 when bitsandbytes is available, which is what makes a teacher and a student fit
    on one card at the same time. Falls back to ``dtype`` with a warning rather than failing.

    Returns:
        A ``transformers`` causal LM in eval mode with gradients disabled.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise DonorError("transformers is required to load a teacher; install `recreator[train]`.") from exc

    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device_map, "revision": revision}
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                # Quantising the quantisation constants too; worth ~0.4 bits per weight.
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            import warnings

            warnings.warn(
                "bitsandbytes is not installed, so the teacher is loading unquantised. "
                "This needs roughly 4x the memory; install bitsandbytes to avoid it.",
                RuntimeWarning,
                stacklevel=2,
            )

    teacher = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher
