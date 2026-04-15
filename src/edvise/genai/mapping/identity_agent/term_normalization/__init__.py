"""
Term column normalization for IdentityAgent: ``term_config`` models, term-stage prompts, and apply helpers.

``TermOrderConfig`` is embedded in term-stage :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermContract`
as ``term_config`` (grain-stage :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.GrainContract` is grain-only).
:func:`apply_term_order_from_config` is exposed lazily via :func:`__getattr__`
so importing this package does not load ``edvise.feature_generation``. The implementation lives in
:mod:`~edvise.genai.mapping.identity_agent.term_normalization.term_order`. Term-stage prompt symbols are also
lazy-loaded from :mod:`edvise.genai.mapping.identity_agent.term_normalization.prompt_builder`.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    CANONICAL_SEASONS,
    InstitutionTermContract,
    SeasonMapEntry,
    TermContract,
    TermOrderConfig,
)

_PROMPT_EXPORTS = frozenset(
    {
        "TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT",
        "TERM_NORMALIZATION_SYSTEM_PROMPT",
        "TERM_NORMALIZATION_USER_TEMPLATE",
        "build_term_normalization_batch_system_prompt",
        "build_term_normalization_batch_user_message_from_grain_and_profiles",
        "build_term_normalization_batch_user_payload",
        "build_term_normalization_system_prompt",
        "build_term_normalization_user_message",
        "build_term_normalization_user_message_from_profiles",
        "parse_institution_term_contracts",
        "parse_institution_term_contracts_with_hitl",
        "parse_term_normalization_pass_output",
        "strip_json_fences",
    }
)

__all__ = [
    "CANONICAL_SEASONS",
    "InstitutionTermContract",
    "TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT",
    "TERM_NORMALIZATION_SYSTEM_PROMPT",
    "TERM_NORMALIZATION_USER_TEMPLATE",
    "SeasonMapEntry",
    "TermContract",
    "TermOrderConfig",
    "apply_term_order_from_config",
    "load_term_extractors_from_hook_spec",
    "term_order_column_for_clean_dataset",
    "term_order_fn_from_term_order_config",
    "build_term_normalization_batch_system_prompt",
    "build_term_normalization_batch_user_message_from_grain_and_profiles",
    "build_term_normalization_batch_user_payload",
    "build_term_normalization_system_prompt",
    "build_term_normalization_user_message",
    "build_term_normalization_user_message_from_profiles",
    "parse_institution_term_contracts",
    "parse_institution_term_contracts_with_hitl",
    "parse_term_normalization_pass_output",
    "strip_json_fences",
    "term_components_to_datetime",
    "term_components_to_datetime_from_series",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from .term_order import apply_term_order_from_config as fn

        return fn
    if name == "load_term_extractors_from_hook_spec":
        from .term_order import load_term_extractors_from_hook_spec as fn

        return fn
    if name == "term_order_column_for_clean_dataset":
        from .term_order import term_order_column_for_clean_dataset as fn

        return fn
    if name == "term_order_fn_from_term_order_config":
        from .term_order import term_order_fn_from_term_order_config as fn

        return fn
    if name == "term_components_to_datetime":
        from .term_datetime import term_components_to_datetime as fn

        return fn
    if name == "term_components_to_datetime_from_series":
        from .term_datetime import term_components_to_datetime_from_series as fn

        return fn
    if name in _PROMPT_EXPORTS:
        from . import prompt_builder as pb

        return getattr(pb, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
