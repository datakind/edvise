"""
Step 2 — Grain contract: prompts, validated LLM output schema, optional row dedupe helpers.

Use with Step 1 output from ``edvise.genai.identity_agent.profiling`` (`KeyProfile`).
"""

from . import deduplication
from .prompt_builder import (
    IDENTITY_AGENT_SYSTEM_PROMPT,
    IDENTITY_AGENT_USER_TEMPLATE,
    build_identity_agent_system_prompt,
    build_identity_agent_user_message,
    format_column_list,
    parse_identity_grain_contract,
    strip_json_fences,
)
from .schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    DedupPolicy,
    DedupStrategy,
    IdentityGrainContract,
)

__all__ = [
    "IDENTITY_CONFIDENCE_HITL_THRESHOLD",
    "DedupPolicy",
    "DedupStrategy",
    "IDENTITY_AGENT_SYSTEM_PROMPT",
    "IDENTITY_AGENT_USER_TEMPLATE",
    "IdentityGrainContract",
    "build_identity_agent_system_prompt",
    "build_identity_agent_user_message",
    "deduplication",
    "format_column_list",
    "parse_identity_grain_contract",
    "strip_json_fences",
]
