"""
Identity pipeline: profiling → grain contract → execution (transforms + schema contract build).

- ``profiling``: deterministic candidate keys and variance facts (`KeyProfile`).
- ``grain_contract``: prompts, `IdentityGrainContract`, optional row helpers.
- ``execution``: grain dedup/term transforms, merge into school config, frozen schema contract.
"""

from . import execution, grain_contract, profiling
from .grain_contract import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    Confidence,
    DedupPolicy,
    DedupStrategy,
    IdentityGrainContract,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    deduplication,
    format_column_list,
    parse_identity_grain_contract,
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
    build_dedupe_fn_from_grain_contract,
    build_schema_contract_from_grain_contracts,
    merge_grain_contracts_into_school_config,
)

__all__ = [
    "CandidateKey",
    "CandidateKeyProfile",
    "ColumnVarianceProfile",
    "Confidence",
    "DedupPolicy",
    "DedupStrategy",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "IdentityGrainContract",
    "KeyProfile",
    "apply_grain_dedup",
    "apply_grain_execution",
    "apply_grain_term_order",
    "build_dedupe_fn_from_grain_contract",
    "build_schema_contract_from_grain_contracts",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "deduplication",
    "execution",
    "format_column_list",
    "grain_contract",
    "merge_grain_contracts_into_school_config",
    "parse_identity_grain_contract",
    "profile_candidate_keys",
    "profiling",
    "strip_json_fences",
]
