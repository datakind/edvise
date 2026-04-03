"""
Step 2 — Grain contract: prompts, validated LLM output schema, optional row dedupe helpers.

Use with Step 1 output from ``edvise.genai.identity_agent.profiling`` (`KeyProfile`).
"""

from . import deduplication
from .prompt_builder import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    TERM_UTILITY_REGISTRY,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    format_column_list,
    parse_identity_grain_contract,
    parse_institution_grain_contracts,
    strip_json_fences,
)
from .runner import run_identity_agent, run_identity_agents_for_institution
from .schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    IdentityGrainContract,
    InstitutionGrainContracts,
    TermFormat,
    TermOrderConfig,
    TermOrderOutputs,
    build_institution_grain_contracts,
)

__all__ = [
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "DedupPolicy",
    "DedupStrategy",
    "TERM_UTILITY_REGISTRY",
    "TermFormat",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "IdentityGrainContract",
    "InstitutionGrainContracts",
    "TermOrderConfig",
    "TermOrderOutputs",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "deduplication",
    "format_column_list",
    "build_institution_grain_contracts",
    "parse_identity_grain_contract",
    "parse_institution_grain_contracts",
    "run_identity_agent",
    "run_identity_agents_for_institution",
    "strip_json_fences",
]
