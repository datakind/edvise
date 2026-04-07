"""
Identity pipeline: profiling → grain contract → execution (transforms + schema contract build).

- ``profiling``: deterministic candidate keys and variance facts (`RankedCandidateProfiles`).
- ``grain_inference``: prompts, `IdentityGrainContract`, optional row helpers.
- ``term_normalization``: Pass 2 prompts, `term_config` models, and :func:`apply_term_order_from_config`.
- ``execution``: grain dedup/term transforms, merge into school config, frozen schema contract.
"""

from __future__ import annotations

from typing import Any

from . import execution, grain_inference, profiling, term_normalization
from .grain_inference import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    IdentityGrainContract,
    InstitutionGrainContracts,
    build_institution_grain_contracts,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    deduplication,
    format_column_list,
    parse_identity_grain_contract,
    parse_institution_grain_contracts,
    run_identity_agent,
    run_identity_agents_for_institution,
    strip_json_fences,
)
from .term_normalization import (
    TERM_NORMALIZATION_SYSTEM_PROMPT,
    TERM_NORMALIZATION_USER_TEMPLATE,
    TERM_UTILITY_REGISTRY,
    TermFormat,
    TermNormalizationPassOutput,
    TermOrderConfig,
    TermOrderOutputs,
    build_term_normalization_system_prompt,
    build_term_normalization_user_message,
    build_term_normalization_user_message_from_profiles,
    parse_term_normalization_pass_output,
)
from .profiling import (
    CandidateKey,
    CandidateProfile,
    ColumnVarianceProfile,
    KeyProfileResult,
    RankedCandidateProfiles,
    RawColumnProfile,
    RawTableProfile,
    profile_candidate_keys,
)
from .execution import (
    UNIQUE_VALUES_MAX_CARDINALITY,
    apply_grain_dedup,
    apply_grain_execution,
    apply_grain_term_order,
    build_dedupe_fn_from_grain_contract,
    build_schema_contract_from_grain_contracts,
    build_training_example_from_schema_contract,
    merge_grain_contracts_into_school_config,
    merge_grain_student_id_alias_into_school_config,
    process_all_schools,
    process_school_dataset,
    save_enriched_schema_contracts,
)

__all__ = [
    "CandidateKey",
    "CandidateProfile",
    "ColumnVarianceProfile",
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "DedupPolicy",
    "DedupStrategy",
    "TERM_NORMALIZATION_SYSTEM_PROMPT",
    "TERM_NORMALIZATION_USER_TEMPLATE",
    "TERM_UTILITY_REGISTRY",
    "TermFormat",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "IdentityGrainContract",
    "InstitutionGrainContracts",
    "KeyProfileResult",
    "RankedCandidateProfiles",
    "RawColumnProfile",
    "RawTableProfile",
    "TermNormalizationPassOutput",
    "TermOrderConfig",
    "TermOrderOutputs",
    "apply_grain_dedup",
    "apply_grain_execution",
    "apply_grain_term_order",
    "apply_term_order_from_config",
    "build_dedupe_fn_from_grain_contract",
    "build_schema_contract_from_grain_contracts",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "build_term_normalization_system_prompt",
    "build_term_normalization_user_message",
    "build_term_normalization_user_message_from_profiles",
    "deduplication",
    "execution",
    "format_column_list",
    "grain_inference",
    "merge_grain_contracts_into_school_config",
    "merge_grain_student_id_alias_into_school_config",
    "UNIQUE_VALUES_MAX_CARDINALITY",
    "build_training_example_from_schema_contract",
    "process_all_schools",
    "process_school_dataset",
    "save_enriched_schema_contracts",
    "build_institution_grain_contracts",
    "parse_identity_grain_contract",
    "parse_institution_grain_contracts",
    "parse_term_normalization_pass_output",
    "profile_candidate_keys",
    "profiling",
    "term_normalization",
    "run_identity_agent",
    "run_identity_agents_for_institution",
    "strip_json_fences",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from edvise.genai.identity_agent.term_normalization.utilities import (
            apply_term_order_from_config as fn,
        )

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
