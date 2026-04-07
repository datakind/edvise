"""
Term column normalization for IdentityAgent: ``term_config`` models and apply helpers.

``TermOrderConfig`` is embedded in :class:`~edvise.genai.identity_agent.grain_inference.schemas.IdentityGrainContract`
as ``term_config``. :func:`apply_term_order_from_config` is exposed lazily via :func:`__getattr__`
so importing this package does not load ``edvise.feature_generation``.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    TERM_UTILITY_REGISTRY,
    TermFormat,
    TermOrderConfig,
    TermOrderOutputs,
)

__all__ = [
    "TERM_UTILITY_REGISTRY",
    "TermFormat",
    "TermOrderConfig",
    "TermOrderOutputs",
    "apply_term_order_from_config",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from .utilities import apply_term_order_from_config as fn

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
