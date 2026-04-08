"""Human-in-the-loop: :class:`HITLItem` schemas, file persistence, and resolver."""

from __future__ import annotations

from edvise.genai.identity_agent.hitl.persistence import (
    build_grain_config_for_resolver,
    build_term_config_for_resolver,
    dedupe_hitl_items,
    write_identity_grain_artifacts,
    write_identity_term_artifacts,
)
from edvise.genai.identity_agent.hitl.resolver import (
    HITLBlockingError,
    HITLValidationError,
    HookValidationError,
    apply_hook_spec,
    check_gate,
    get_hook_items,
    resolve_items,
    validate_hook,
)
from edvise.genai.identity_agent.hitl.schemas import (
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
    "ReentryDepth",
    "TermResolution",
    "apply_hook_spec",
    "build_grain_config_for_resolver",
    "build_term_config_for_resolver",
    "check_gate",
    "dedupe_hitl_items",
    "get_hook_items",
    "resolve_items",
    "validate_hook",
    "write_identity_grain_artifacts",
    "write_identity_term_artifacts",
]
