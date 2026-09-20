"""Exception types raised by recreator."""

from __future__ import annotations


class RecreatorError(Exception):
    """Base class for every error recreator raises on purpose."""


class DonorError(RecreatorError):
    """The donor checkpoint could not be read, or its architecture is unsupported."""


class GraftError(RecreatorError):
    """A tensor could not be transferred from donor to student."""


class PlanError(RecreatorError):
    """The requested recreation does not fit the declared hardware budget."""
