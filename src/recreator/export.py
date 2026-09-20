"""Saving a recreated model and publishing it.

A HELIX model is not a ``transformers`` architecture, so a checkpoint has to carry enough for
someone else to load it: the HELIX config and weights, the donor's tokenizer (the student inherited
the donor's vocabulary, so it is the donor's tokenizer that makes its ids mean anything), and a
model card saying what it was made from and what was actually trained.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .errors import RecreatorError

_LOAD_SNIPPET = """```python
from helix_lm import HelixForCausalLM
from transformers import AutoTokenizer

model = HelixForCausalLM.from_pretrained("{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
print(tokenizer.decode(model.generate(ids, max_new_tokens=32, temperature=0.7)[0]))
```"""


def model_card(
    *,
    repo: str,
    donor_id: str,
    config: Any,
    report: Any = None,
    record: Any = None,
    trainable_fraction: float | None = None,
) -> str:
    """Write a model card describing how the model was recreated."""
    lines = [
        "---",
        "library_name: helix-lm",
        "tags:",
        "  - helix",
        "  - recreator",
        "  - distillation",
        f"base_model: {donor_id}",
        "---",
        "",
        f"# {repo.split('/')[-1]}",
        "",
        f"A [HELIX](https://pypi.org/project/helix-lm/) model recreated from `{donor_id}` with "
        "[recreator](https://pypi.org/project/recreator/).",
        "",
        "The donor's attention, MLP, norms and embeddings were grafted directly into a HELIX",
        "student. The strands HELIX adds -- a gated delta-rule recurrence and a hierarchical",
        "landmark index -- had no donor counterpart, so they were initialised silent and trained",
        "against the donor's own outputs.",
        "",
        "## Architecture",
        "",
        "| | |",
        "| --- | --- |",
        f"| hidden size | {config.hidden_size} |",
        f"| layers | {config.num_hidden_layers} |",
        f"| attention heads | {config.num_attention_heads} (kv {config.num_key_value_heads}) |",
        f"| local span | {config.local_span} tokens |",
        f"| index top-k | {config.index_topk} blocks |",
        f"| max position | {config.max_position_embeddings:,} |",
        "",
    ]
    if report is not None:
        lines += ["## Graft", "", f"- {report.summary()}", ""]
    if trainable_fraction is not None:
        lines.append(f"- trained {trainable_fraction:.1%} of parameters; the rest came from the donor\n")
    if record is not None and getattr(record, "losses", None):
        lines += [
            "## Training",
            "",
            f"- {record.steps} steps",
            f"- final loss {record.final_loss:.4f}" if record.final_loss is not None else "",
            "",
        ]
    lines += ["## Usage", "", _LOAD_SNIPPET.format(repo=repo), ""]
    return "\n".join(line for line in lines if line is not None)


def save_recreated(
    model,
    directory: str | Path,
    *,
    donor_path: str | Path | None = None,
    donor_id: str = "unknown",
    report: Any = None,
    record: Any = None,
    repo: str | None = None,
    trainable_fraction: float | None = None,
) -> Path:
    """Write a recreated model, its tokenizer and a model card to ``directory``.

    Args:
        model: The recreated ``HelixForCausalLM``.
        directory: Destination.
        donor_path: Local donor checkpoint, whose tokenizer files are copied across.
        donor_id: Donor name for the card.
        report: The :class:`~recreator.mapping.GraftReport`, recorded alongside.
        record: The :class:`~recreator.distill.TrainingRecord`, recorded alongside.
        repo: Repo name used in the card's usage snippet.
        trainable_fraction: Share of parameters trained.

    Returns:
        The directory written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)

    if donor_path is not None:
        donor_path = Path(donor_path)
        for pattern in ("tokenizer*", "*.model", "special_tokens_map.json", "vocab*", "merges.txt", "chat_template*"):
            for source in donor_path.glob(pattern):
                if source.is_file():
                    shutil.copy2(source, directory / source.name)

    provenance: dict[str, Any] = {"donor": donor_id, "recreator_version": _version()}
    if report is not None:
        provenance["graft"] = {
            "summary": report.summary(),
            "coverage": report.coverage,
            "transferred_parameters": report.transferred_parameters,
            "student_parameters": report.student_parameters,
            "adapted": [asdict(entry) for entry in report.transferred if entry.adapted],
            "left_at_init": report.missing,
        }
    if record is not None:
        provenance["training"] = {
            "steps": record.steps,
            "final_loss": record.final_loss,
            "sequence_lengths": sorted(set(record.sequence_lengths)),
        }
    (directory / "recreator.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")

    (directory / "README.md").write_text(
        model_card(
            repo=repo or directory.name,
            donor_id=donor_id,
            config=model.config,
            report=report,
            record=record,
            trainable_fraction=trainable_fraction,
        )
    )
    return directory


def _version() -> str:
    from ._version import __version__

    return __version__


def push_to_hub(
    directory: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    commit_message: str = "Upload model recreated with recreator",
) -> str:
    """Upload a saved recreation to the Hugging Face Hub.

    Args:
        directory: A directory written by :func:`save_recreated`.
        repo_id: Target repo, ``"user/name"``.
        private: Create the repo private. Defaults to private, since a recreation inherits its
            donor's licence and that is the owner's call to make, not this tool's.
        token: Hub token; falls back to the ambient login.
        commit_message: Commit message.

    Returns:
        The repo URL.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise RecreatorError("huggingface_hub is required to push; install `recreator[hub]`.") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(directory), repo_id=repo_id, repo_type="model", commit_message=commit_message
    )
    return f"https://huggingface.co/{repo_id}"
