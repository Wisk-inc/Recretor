"""The techniques a recreation is built from.

Each module here addresses one reason a naive architecture swap fails:

* :mod:`~recreator.techniques.gates` -- the new strands would corrupt the grafted weights, so they
  are made to contribute exactly zero at step zero while staying trainable.
* :mod:`~recreator.techniques.expert_merge` -- a MoE donor has no dense MLP to copy, so its experts
  are concatenated into one that reproduces the routed mixture exactly.
* :mod:`~recreator.techniques.teacher_cache` -- the teacher is frozen, so it need not be resident;
  its top-k outputs are precomputed to disk and replayed.
* :mod:`~recreator.techniques.schedules` -- strand I has nothing to retrieve at short context, so
  sequence length climbs a ladder instead of starting at its final value.
"""

from .gates import (
    DEFAULT_GATE_BIAS,
    BiasedStrandGate,
    anneal_gate_bias,
    gate_report,
    install_strand_gates,
    silence_recurrent_strands,
    strand_parameter_names,
)
from .expert_merge import ExpertStack, collect_experts, concat_experts, merge_experts, router_prior
from .schedules import ContextLadder, cosine_with_warmup
from .teacher_cache import TeacherCache, sparse_kl_loss

__all__ = [
    "DEFAULT_GATE_BIAS",
    "BiasedStrandGate",
    "ContextLadder",
    "ExpertStack",
    "TeacherCache",
    "anneal_gate_bias",
    "collect_experts",
    "concat_experts",
    "cosine_with_warmup",
    "gate_report",
    "install_strand_gates",
    "merge_experts",
    "router_prior",
    "silence_recurrent_strands",
    "sparse_kl_loss",
    "strand_parameter_names",
]
