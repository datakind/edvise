"""
Identity pipeline: profiling → grain contract → execution (transforms + schema contract build).

- ``profiling``: deterministic candidate keys and variance facts (`KeyProfile`).
- ``grain_inference``: prompts, `IdentityGrainContract`, optional row helpers.
- ``execution``: grain dedup/term transforms, merge into school config, frozen schema contract.
"""

from . import execution, grain_inference, profiling
from .grain_inference import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    IdentityGrainContract,
    InstitutionGrainContracts,
    TermOrderConfig,
    TermOrderOutputs,
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
from .profiling import (
    CandidateKey,
    CandidateKeyProfile,
    ColumnVarianceProfile,
    KeyProfile,
    profile_candidate_keys,
)
from .execution import (
    apply_grain_dedup,
    apply_grain_execution,
    apply_grain_term_order,
    apply_term_order_from_config,
    build_dedupe_fn_from_grain_contract,
    build_schema_contract_from_grain_contracts,
    merge_grain_contracts_into_school_config,
)

__all__ = [
    "CandidateKey",
    "CandidateKeyProfile",
    "ColumnVarianceProfile",
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "DedupPolicy",
    "DedupStrategy",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "IdentityGrainContract",
    "InstitutionGrainContracts",
    "KeyProfile",
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
    "deduplication",
    "execution",
    "format_column_list",
    "grain_inference",
    "merge_grain_contracts_into_school_config",
    "build_institution_grain_contracts",
    "parse_identity_grain_contract",
    "parse_institution_grain_contracts",
    "profile_candidate_keys",
    "profiling",
    "run_identity_agent",
    "run_identity_agents_for_institution",
    "strip_json_fences",
]
