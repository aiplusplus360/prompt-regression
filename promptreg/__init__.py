"""Regression testing for prompts.

A prompt is a dependency. It has inputs, outputs, and it can break a product
when it changes. This package treats prompts the way a test suite treats code:
you record expected behaviour, run it on every change, and fail the build when
it drifts.
"""

__version__ = "0.1.0"

from .diff import CompareResult, compare
from .providers import EchoProvider, OpenAICompatibleProvider, Provider, get_provider
from .suite import Case, Suite, load_suite

__all__ = [
    "Case",
    "CompareResult",
    "EchoProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "Suite",
    "compare",
    "get_provider",
    "load_suite",
]
