"""
Step 2 — Grain contract: prompts, validated LLM output schema, optional row dedupe helpers.

Use with Step 1 output from ``edvise.genai.identity_agent.profiling`` (`RankedCandidateProfiles`).
"""

from . import deduplication
from .prompt_builder import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    format_column_list,
    parse_grain_contract,
    parse_grain_contract_with_hitl,
    parse_institution_grain_contracts,
    strip_json_fences,
)
from .runner import (
    run_identity_agent,
    run_identity_agent_with_hitl,
    run_identity_agents_for_institution,
    run_identity_agents_for_institution_with_hitl,
)
from .schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    GrainContract,
    InstitutionGrainContract,
    build_institution_grain_contracts,
)

__all__ = [
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "DedupPolicy",
    "DedupStrategy",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "GrainContract",
    "InstitutionGrainContract",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "deduplication",
    "format_column_list",
    "build_institution_grain_contracts",
    "parse_grain_contract",
    "parse_grain_contract_with_hitl",
    "parse_institution_grain_contracts",
    "run_identity_agent",
    "run_identity_agent_with_hitl",
    "run_identity_agents_for_institution",
    "run_identity_agents_for_institution_with_hitl",
    "strip_json_fences",
]
