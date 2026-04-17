"""Canonical relative paths for materialized hook modules (not LLM-chosen)."""

from __future__ import annotations

from pathlib import Path

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain


def default_hook_module_relpath(institution_id: str, domain: HITLDomain) -> str:
    """
    Stable path relative to the **hook modules root** passed as ``repo_root`` / ``hook_file_root``
    (in ``ia_dev`` this is ``{bronze_volumes_path}/identity_agent``, not the bare bronze root).

    Layout: ``identity_hooks/<institution_id>/dedup_hooks.py`` (grain) or ``term_hooks.py`` (term).
    Grain and term use distinct filenames so one institution can have both modules.
    """
    if domain == HITLDomain.IDENTITY_GRAIN:
        basename = "dedup_hooks.py"
    elif domain == HITLDomain.IDENTITY_TERM:
        basename = "term_hooks.py"
    else:
        raise ValueError(
            f"Hook modules are only defined for grain and term, not {domain!r}"
        )
    return f"identity_hooks/{institution_id}/{basename}"


def resolve_hook_module_path(file_relpath: str, *, root: str | Path) -> Path:
    """
    Resolve :attr:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.HookSpec.file`
    under ``root`` (materialize / validate_hook / runtime import).

    Raises ``ValueError`` if ``file_relpath`` is absolute or escapes ``root``.
    """
    base = Path(root).resolve()
    rel = Path(file_relpath)
    if rel.is_absolute():
        raise ValueError(f"hook_spec.file must be relative, got {file_relpath!r}")
    out = (base / rel).resolve()
    try:
        out.relative_to(base)
    except ValueError as e:
        raise ValueError(f"hook_spec.file {file_relpath!r} escapes root {base}") from e
    return out


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


__all__ = [
    "default_hook_module_relpath",
    "ensure_hook_spec_file",
    "resolve_hook_module_path",
]
