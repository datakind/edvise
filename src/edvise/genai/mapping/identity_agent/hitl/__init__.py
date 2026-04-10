"""Human-in-the-loop: :class:`HITLItem` schemas, on-disk artifacts, and resolver."""

from __future__ import annotations

from edvise.genai.mapping.identity_agent.hitl.hook_generation import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
    generate_hook_spec,
    generate_hook_specs_for_hook_items,
    parse_hook_spec,
)
from edvise.genai.mapping.identity_agent.hitl.artifacts import (
    build_grain_config_for_resolver,
    build_term_config_for_resolver,
    unique_hitl_items_by_item_id,
    write_identity_grain_artifacts,
    write_identity_term_artifacts,
)
from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLBlockingError,
    HITLValidationError,
    HookValidationError,
    apply_hook_spec,
    check_gate,
    get_hook_items,
    resolve_items,
    validate_hook,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLResolution,
    HITLStatus,
    HITLTarget,
    InstitutionHITLItems,
    ReentryDepth,
    TermResolution,
)

__all__ = [
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "extract_config_snippet_for_hook_item",
    "generate_hook_spec",
    "generate_hook_specs_for_hook_items",
    "GrainResolution",
    "HITLBlockingError",
    "HITLDomain",
    "HITLItem",
    "HITLOption",
    "HITLResolution",
    "HITLStatus",
    "HITLTarget",
    "HITLValidationError",
    "HookValidationError",
    "InstitutionHITLItems",
    "parse_hook_spec",
    "ReentryDepth",
    "TermResolution",
    "apply_hook_spec",
    "build_grain_config_for_resolver",
    "build_term_config_for_resolver",
    "check_gate",
    "unique_hitl_items_by_item_id",
    "get_hook_items",
    "resolve_items",
    "validate_hook",
    "write_identity_grain_artifacts",
    "write_identity_term_artifacts",
]
