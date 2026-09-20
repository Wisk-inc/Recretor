"""recreator -- rebuild a pretrained transformer as a HELIX model without pretraining it again.

A donor transformer and a HELIX block share most of their parameters: the same attention
projections, the same per-head norms, the same SwiGLU MLP, the same residual norms, the same
embeddings. What HELIX adds -- a gated delta-rule recurrence and a hierarchical landmark index --
has no counterpart in the donor, and that is the only part a recreation has to learn.

So a recreation is a graft plus a repair:

    >>> from recreator import recreate                       # doctest: +SKIP
    >>> result = recreate("Qwen/Qwen3-8B", preset="balanced")  # doctest: +SKIP

The graft copies every shared tensor and then makes the new strands contribute *exactly zero*, so
the student's first forward pass reproduces the donor's within its local window -- verifiably, to
floating-point noise. Training starts from the donor's loss rather than from noise, and touches
only the parameters the donor could not supply.

See :mod:`recreator.techniques` for the individual mechanisms and :mod:`recreator.planner` for
working out what fits on a given card before starting.
"""

from ._version import __version__
from .distill import DistillConfig, TrainingRecord, distill, distill_layerwise, freeze_to_new_strands
from .donor import DonorHandle, load_teacher, resolve_donor
from .errors import DonorError, GraftError, PlanError, RecreatorError
from .export import model_card, push_to_hub, save_recreated
from .graft import GraftResult, graft
from .mapping import FitReport, GraftReport
from .planner import MemoryPlan, count_parameters, plan_memory
from .student import PRESETS, Preset, derive_student_config
from .techniques import ContextLadder, TeacherCache

__all__ = [
    "PRESETS",
    "ContextLadder",
    "DistillConfig",
    "DonorError",
    "DonorHandle",
    "FitReport",
    "GraftError",
    "GraftReport",
    "GraftResult",
    "MemoryPlan",
    "PlanError",
    "Preset",
    "RecreatorError",
    "TeacherCache",
    "TrainingRecord",
    "__version__",
    "count_parameters",
    "derive_student_config",
    "distill",
    "distill_layerwise",
    "freeze_to_new_strands",
    "graft",
    "load_teacher",
    "model_card",
    "plan_memory",
    "push_to_hub",
    "recreate",
    "resolve_donor",
    "save_recreated",
]


def recreate(donor: str, **kwargs):
    """Graft a donor into a HELIX student. Convenience wrapper around :func:`recreator.graft.graft`."""
    return graft(donor, **kwargs)
