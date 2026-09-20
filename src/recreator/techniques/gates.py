"""Strand gates: the mechanism that lets a grafted model start as its donor.

A HELIX block braids three sequence mixers into one residual stream. Two of them -- the recurrent
strand R and the landmark index strand I -- have no counterpart in a plain transformer, so a graft
has nothing to put in them. Left at their random initialisation they do not merely fail to help;
they inject noise into a residual stream that the donor's own weights were tuned against, and the
grafted model scores worse than the donor it was built from.

The fix is to make both new strands contribute *exactly zero* at step zero, while keeping a live
gradient path so training can bring them in:

* **Strand R** ends in ``out_proj``. Zeroing that one matrix makes the strand's contribution
  identically zero, and because its input is not zero the gradient with respect to ``out_proj``
  stays non-zero -- the strand learns its way back in rather than being cut off. This is the
  standard zero-initialised residual branch.

* **Strand I** is not additive. It is mixed with strand L through a two-way softmax::

      gate = strand_gate(h).view(..., heads, 2).softmax(-1)
      out  = gate[..., :1] * local + gate[..., 1:] * index

  Zeroing ``strand_gate`` therefore gives ``softmax([0, 0]) == [0.5, 0.5]`` -- a half-and-half blend
  of the donor's attention with an untrained index, which is the opposite of what we want. HELIX
  builds ``strand_gate`` without a bias, so there is no constant to lean on either. We swap in a
  bias-carrying ``nn.Linear`` of the same shape, zero the weight and set the bias to
  ``[+b, -b]`` per head. The gate then starts at ``sigmoid(2b)`` on strand L -- 0.9999 at the default
  ``b = 4.5`` -- and, being an ordinary parameter, is free to open the index as training proceeds.

The result is a student whose forward pass at step zero is the donor's, restricted to HELIX's local
attention window, with the new machinery present, silent and trainable.
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: Logit half-gap applied to a freshly grafted strand-I gate. ``softmax([b, -b])`` puts
#: ``sigmoid(2b) ~= 0.99988`` of the mix on strand L, which is silent to about four decimal places
#: while staying far away from the saturated region where gradients vanish.
DEFAULT_GATE_BIAS = 4.5


class BiasedStrandGate(nn.Linear):
    """A ``strand_gate`` that can be biased towards one strand at initialisation.

    Shape-compatible with the ``nn.Linear(hidden, heads * 2, bias=False)`` HELIX builds, so it can be
    swapped in without touching the block's forward pass.
    """

    def __init__(self, in_features: int, out_features: int, *, gate_bias: float = DEFAULT_GATE_BIAS) -> None:
        super().__init__(in_features, out_features, bias=True)
        self.gate_bias = float(gate_bias)
        self.reset_to_local()

    def reset_to_local(self) -> None:
        """Silence strand I: zero weight, bias ``[+b, -b]`` per head."""
        with torch.no_grad():
            self.weight.zero_()
            bias = self.bias.view(-1, 2)
            bias[:, 0] = self.gate_bias
            bias[:, 1] = -self.gate_bias

    @property
    def index_share(self) -> torch.Tensor:
        """Per-head weight currently placed on strand I, ignoring the input-dependent term."""
        bias = self.bias.detach().view(-1, 2)
        return torch.softmax(bias, dim=-1)[:, 1]


def install_strand_gates(model: nn.Module, *, gate_bias: float = DEFAULT_GATE_BIAS) -> list[str]:
    """Replace every ``strand_gate`` in ``model`` with a :class:`BiasedStrandGate`.

    Only layers that actually run strand I carry a ``strand_gate``, so layers scheduled as
    ``helix_local`` are skipped automatically.

    Args:
        model: A ``HelixForCausalLM`` (or any module tree containing HELIX braids).
        gate_bias: Logit half-gap; larger means a more thoroughly silenced index at step zero.

    Returns:
        The qualified names of the gates that were replaced.
    """
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        gate = getattr(module, "strand_gate", None)
        if not isinstance(gate, nn.Linear) or isinstance(gate, BiasedStrandGate):
            continue
        new_gate = BiasedStrandGate(gate.in_features, gate.out_features, gate_bias=gate_bias)
        new_gate.to(device=gate.weight.device, dtype=gate.weight.dtype)
        module.strand_gate = new_gate
        replaced.append(f"{name}.strand_gate")
    return replaced


def silence_recurrent_strands(model: nn.Module) -> list[str]:
    """Zero every recurrent strand's output projection, making strand R contribute nothing.

    Returns:
        The qualified names of the projections that were zeroed.
    """
    zeroed: list[str] = []
    for name, module in model.named_modules():
        if type(module).__name__ != "HelixRecurrentStrand":
            continue
        with torch.no_grad():
            module.out_proj.weight.zero_()
        zeroed.append(f"{name}.out_proj")
    return zeroed


def strand_parameter_names(model: nn.Module) -> list[str]:
    """Names of the parameters that belong to the *new* strands, not to the graft.

    These are what :func:`recreator.distill.freeze_to_new_strands` leaves trainable: everything the
    donor could not supply, and nothing it could.
    """
    names: list[str] = []
    for name, module in model.named_modules():
        kind = type(module).__name__
        is_new = kind == "HelixRecurrentStrand" or isinstance(module, BiasedStrandGate)
        if is_new:
            names.extend(f"{name}.{param}" for param, _ in module.named_parameters(recurse=True))
    # Strand I's routing machinery lives directly on the braid rather than in a module of its own.
    index_only = ("route_q_proj", "leaf_pooler", "node_pooler", "level_bias", "distance_bias")
    for name, module in model.named_modules():
        if type(module).__name__ != "HelixBraid":
            continue
        for attr in index_only:
            target = getattr(module, attr, None)
            if isinstance(target, nn.Module):
                names.extend(f"{name}.{attr}.{p}" for p, _ in target.named_parameters(recurse=True))
            elif isinstance(target, nn.Parameter):
                names.append(f"{name}.{attr}")
    return sorted(set(names))


def gate_report(model: nn.Module) -> dict[str, float]:
    """Mean strand-I share per gated layer, for watching the index open during training."""
    report: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, BiasedStrandGate):
            report[name] = float(module.index_share.mean())
    return report


def anneal_gate_bias(model: nn.Module, fraction: float, *, start: float = DEFAULT_GATE_BIAS) -> None:
    """Optionally relax the initial gate bias over training.

    The gate is a learned parameter and normally needs no help, but on short runs the bias can be
    walked down on a schedule so strand I is *forced* to carry some of the mix before the run ends.

    Args:
        model: The student.
        fraction: Training progress in ``[0, 1]``.
        start: The bias the graft began with.
    """
    scale = 0.5 * (1.0 + math.cos(math.pi * min(max(fraction, 0.0), 1.0)))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BiasedStrandGate):
                bias = module.bias.view(-1, 2)
                bias[:, 0] = start * scale
                bias[:, 1] = -start * scale
