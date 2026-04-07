"""
Term column normalization for IdentityAgent: ``term_config`` models, Pass 2 prompts, and apply helpers.

``TermOrderConfig`` is embedded in :class:`~edvise.genai.identity_agent.grain_inference.schemas.IdentityGrainContract`
as ``term_config``. :func:`apply_term_order_from_config` is exposed lazily via :func:`__getattr__`
so importing this package does not load ``edvise.feature_generation``. Pass 2 prompt symbols are also
lazy-loaded from :mod:`edvise.genai.identity_agent.term_normalization.prompt_builder`.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    TERM_UTILITY_REGISTRY,
    TermFormat,
    TermOrderConfig,
    TermOrderOutputs,
)

_PROMPT_EXPORTS = frozenset(
    {
        "TERM_NORMALIZATION_SYSTEM_PROMPT",
        "TERM_NORMALIZATION_USER_TEMPLATE",
        "TermNormalizationPassOutput",
        "build_term_normalization_system_prompt",
        "build_term_normalization_user_message",
        "build_term_normalization_user_message_from_profiles",
        "parse_term_normalization_pass_output",
        "strip_json_fences",
    }
)

__all__ = [
    "TERM_NORMALIZATION_SYSTEM_PROMPT",
    "TERM_NORMALIZATION_USER_TEMPLATE",
    "TERM_UTILITY_REGISTRY",
    "TermFormat",
    "TermNormalizationPassOutput",
    "TermOrderConfig",
    "TermOrderOutputs",
    "apply_term_order_from_config",
    "build_term_normalization_system_prompt",
    "build_term_normalization_user_message",
    "build_term_normalization_user_message_from_profiles",
    "parse_term_normalization_pass_output",
    "strip_json_fences",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from .utilities import apply_term_order_from_config as fn

        return fn
    if name in _PROMPT_EXPORTS:
        from . import prompt_builder as pb

        return getattr(pb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
