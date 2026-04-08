"""
Identity pipeline: profiling → grain contract → execution (transforms + schema contract build).

- ``profiling``: deterministic candidate keys and variance facts (`RankedCandidateProfiles`).
- ``grain_inference``: prompts, `GrainContract` (pass 1), optional row helpers.
- ``term_normalization``: Pass 2 prompts, `term_config` models, and :func:`apply_term_order_from_config`.
- ``execution``: grain dedup + term contract application, merge into school config; emits the single
  SMA-facing frozen contract type :class:`EnrichedSchemaContractForSMA`.
- ``identity_bundle``: optional combined handoff :class:`InstitutionIdentityContract` (pass 1 + pass 2 envelopes).
"""

from __future__ import annotations

from typing import Any

from . import execution, grain_inference, profiling, term_normalization
from .hitl import (
    check_gate,
    write_identity_grain_artifacts,
    write_identity_term_artifacts,
)
from .grain_inference import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    GrainContract,
    InstitutionGrainContract,
    build_institution_grain_contracts,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    deduplication,
    format_column_list,
    parse_grain_contract,
    parse_grain_contract_with_hitl,
    parse_institution_grain_contracts,
    run_identity_agent,
    run_identity_agent_with_hitl,
    run_identity_agents_for_institution,
    run_identity_agents_for_institution_with_hitl,
    strip_json_fences,
)
from .term_normalization import (
    CANONICAL_SEASONS,
    InstitutionTermContract,
    TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT,
    TERM_NORMALIZATION_SYSTEM_PROMPT,
    TERM_NORMALIZATION_USER_TEMPLATE,
    SeasonMapEntry,
    TermContract,
    TermOrderConfig,
    build_term_normalization_batch_system_prompt,
    build_term_normalization_batch_user_message_from_grain_and_profiles,
    build_term_normalization_batch_user_payload,
    build_term_normalization_system_prompt,
    build_term_normalization_user_message,
    build_term_normalization_user_message_from_profiles,
    parse_institution_term_contracts,
    parse_institution_term_contracts_with_hitl,
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
    apply_term_order_from_contract,
    build_dedupe_fn_from_grain_contract,
    build_enriched_schema_contract_for_dataset,
    build_schema_contract_from_grain_contracts,
    build_training_example_from_schema_contract,
    EnrichedSchemaContractForSMA,
    merge_grain_contracts_into_school_config,
    merge_grain_student_id_alias_into_school_config,
    parse_enriched_schema_contract_for_sma,
    process_school_dataset,
    save_enriched_schema_contract,
    save_enriched_schema_contracts,
)
from .identity_bundle import (
    InstitutionIdentityContract,
    institution_identity_contract_from_parts,
)

__all__ = [
    "CandidateKey",
    "CandidateProfile",
    "ColumnVarianceProfile",
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "DedupPolicy",
    "DedupStrategy",
    "EnrichedSchemaContractForSMA",
    "CANONICAL_SEASONS",
    "InstitutionTermContract",
    "TERM_NORMALIZATION_BATCH_SYSTEM_PROMPT",
    "TERM_NORMALIZATION_SYSTEM_PROMPT",
    "TERM_NORMALIZATION_USER_TEMPLATE",
    "SeasonMapEntry",
    "TermContract",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "GrainContract",
    "InstitutionGrainContract",
    "InstitutionIdentityContract",
    "KeyProfileResult",
    "RankedCandidateProfiles",
    "RawColumnProfile",
    "RawTableProfile",
    "TermOrderConfig",
    "apply_grain_dedup",
    "apply_grain_execution",
    "apply_term_order_from_contract",
    "apply_term_order_from_config",
    "build_dedupe_fn_from_grain_contract",
    "build_enriched_schema_contract_for_dataset",
    "build_schema_contract_from_grain_contracts",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "build_term_normalization_batch_system_prompt",
    "build_term_normalization_batch_user_message_from_grain_and_profiles",
    "build_term_normalization_batch_user_payload",
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
    "process_school_dataset",
    "save_enriched_schema_contract",
    "save_enriched_schema_contracts",
    "build_institution_grain_contracts",
    "check_gate",
    "institution_identity_contract_from_parts",
    "parse_grain_contract",
    "parse_grain_contract_with_hitl",
    "parse_institution_grain_contracts",
    "parse_enriched_schema_contract_for_sma",
    "parse_institution_term_contracts",
    "parse_institution_term_contracts_with_hitl",
    "parse_term_normalization_pass_output",
    "profile_candidate_keys",
    "profiling",
    "term_normalization",
    "run_identity_agent",
    "run_identity_agent_with_hitl",
    "run_identity_agents_for_institution",
    "run_identity_agents_for_institution_with_hitl",
    "strip_json_fences",
    "write_identity_grain_artifacts",
    "write_identity_term_artifacts",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from edvise.genai.identity_agent.term_normalization.utilities import (
            apply_term_order_from_config as fn,
        )

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
