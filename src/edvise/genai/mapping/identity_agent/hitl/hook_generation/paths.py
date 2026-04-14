"""Canonical relative paths for materialized hook modules (not LLM-chosen)."""

from __future__ import annotations

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain


def default_hook_module_relpath(institution_id: str, domain: HITLDomain) -> str:
    """
    Stable path under the repo root: ``pipelines/<institution_id>/helpers/<module>.py``.

    Grain and term use distinct filenames so one institution can have both modules.
    """
    if domain == HITLDomain.IDENTITY_GRAIN:
        basename = "dedup_hooks.py"
    elif domain == HITLDomain.IDENTITY_TERM:
        basename = "term_hooks.py"
    else:
        raise ValueError(f"Hook modules are only defined for grain and term, not {domain!r}")
    return f"pipelines/{institution_id}/helpers/{basename}"


def ensure_hook_spec_file(
    hook_spec: HookSpec,
    *,
    institution_id: str,
    domain: HITLDomain,
) -> HookSpec:
    """
    Set ``hook_spec.file`` to the canonical path for this institution and domain.

    Always overwrites ``file`` so stale or model-supplied paths cannot leak into config.
    """
    return hook_spec.model_copy(
        update={"file": default_hook_module_relpath(institution_id, domain)}
    )


__all__ = ["default_hook_module_relpath", "ensure_hook_spec_file"]
