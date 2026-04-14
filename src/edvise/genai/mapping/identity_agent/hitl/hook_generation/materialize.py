"""
Materialize :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.HookSpec` as importable Python modules.

Domain-agnostic: each ``HookFunctionSpec.draft`` is written verbatim (full ``def`` blocks).
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Sequence
from pathlib import Path

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookFunctionSpec,
    HookSpec,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    resolve_hook_module_path,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain

logger = logging.getLogger(__name__)


def merge_hook_specs(
    *specs: HookSpec,
    repo_root: str | Path | None = None,
) -> HookSpec:
    """
    Combine multiple :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.HookSpec`
    instances that target the **same** materialized module path.

    Use this when different HITL resolutions (e.g. opaque numeric term + date-string term) each
    contribute a pair of functions but all use ``identity_hooks/<institution_id>/term_hooks.py``.
    :func:`materialize_hook_spec_to_file` overwrites the file — merge first, then materialize once.

    All specs must share one output file:

    - With ``repo_root``: resolved paths (via :func:`~edvise.genai.mapping.identity_agent.hitl.hook_generation.paths.resolve_hook_module_path`) must be identical.
    - Without ``repo_root``: ``hook_spec.file`` strings must be identical and non-null.

    Duplicate function ``name`` values are allowed when the ``draft`` text matches (same
    definition from multiple tables); only the first copy is kept. Conflicting drafts for the
    same name raise.

    The first spec's ``file`` is kept on the result.
    """
    from edvise.genai.mapping.identity_agent.hitl.resolver import HITLValidationError

    if not specs:
        raise HITLValidationError("merge_hook_specs requires at least one HookSpec")
    for s in specs:
        if not s.file:
            raise HITLValidationError(
                "merge_hook_specs: every HookSpec must set file — "
                "run ensure_hook_spec_file / apply_hook_spec before merge."
            )

    if repo_root is not None:
        roots = {
            resolve_hook_module_path(s.file, root=repo_root).resolve() for s in specs
        }
        if len(roots) != 1:
            raise HITLValidationError(
                "merge_hook_specs: all HookSpec.file paths must resolve to the same location "
                f"under repo_root; got {roots!r}"
            )
    else:
        files = {s.file for s in specs}
        if len(files) != 1:
            raise HITLValidationError(
                "merge_hook_specs: all HookSpec.file values must match when repo_root is omitted; "
                f"got {files!r}"
            )

    merged_functions: list[HookFunctionSpec] = []
    seen_draft_by_name: dict[str, str] = {}
    for s in specs:
        for fn in s.functions:
            draft = (fn.draft or "").strip()
            if fn.name in seen_draft_by_name:
                if seen_draft_by_name[fn.name] != draft:
                    raise HITLValidationError(
                        f"merge_hook_specs: duplicate function name {fn.name!r} with conflicting "
                        "drafts — use one definition or align drafts across HookSpecs."
                    )
                continue
            seen_draft_by_name[fn.name] = draft
            merged_functions.append(fn)

    return HookSpec(file=specs[0].file, functions=merged_functions)


def materialize_hook_specs_to_file(
    specs: Sequence[HookSpec],
    *,
    repo_root: str | Path,
    domain: HITLDomain | None = None,
) -> Path:
    """
    Merge ``specs`` with :func:`merge_hook_specs` (same rules: one file, unique names), then write.

    Equivalent to ``materialize_hook_spec_to_file(merge_hook_specs(...), ...)`` when multiple
    resolutions contribute functions to a single module.
    """
    if not specs:
        from edvise.genai.mapping.identity_agent.hitl.resolver import (
            HITLValidationError,
        )

        raise HITLValidationError(
            "materialize_hook_specs_to_file requires at least one HookSpec"
        )
    merged = merge_hook_specs(*specs, repo_root=repo_root)
    return materialize_hook_spec_to_file(merged, repo_root=repo_root, domain=domain)


def materialize_hook_spec_to_file(
    hook_spec: HookSpec,
    *,
    repo_root: str | Path,
    domain: HITLDomain | None = None,
) -> Path:
    """
    Write ``hook_spec.file`` under ``repo_root`` by concatenating each function ``draft`` verbatim.

    ``repo_root`` should be the school's bronze volume root (``bronze_volumes_path`` — the same
    base used for ``cleaned/`` and ``enriched_schema_contracts/``), or any directory you treat as
    the root for relative :attr:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.HookSpec.file` paths.

    Validates the assembled module with :func:`ast.parse` before writing. Optionally runs pyflakes
    (warning only). Draft vs runtime **signature** checks run in
    :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.validate_hook` after import.

    ``domain`` is optional; when set, it is recorded in the module header comment only.
    """
    from edvise.genai.mapping.identity_agent.hitl.resolver import HITLValidationError

    if not hook_spec.file:
        raise HITLValidationError(
            "hook_spec.file is null — run ensure_hook_spec_file / apply_hook_spec before materialize."
        )

    try:
        out_path = resolve_hook_module_path(hook_spec.file, root=repo_root)
    except ValueError as e:
        raise HITLValidationError(str(e)) from e
    repo_root = Path(repo_root).resolve()

    text = _assemble_module_text(hook_spec, domain=domain)
    _validate_module_ast(text, out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"✓ Hook module written to {out_path.relative_to(repo_root)}")

    _maybe_run_pyflakes(text, out_path)

    return out_path


def _assemble_module_text(hook_spec: HookSpec, *, domain: HITLDomain | None) -> str:
    from edvise.genai.mapping.identity_agent.hitl.resolver import HITLValidationError

    parts: list[str] = [
        "# Generated by edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize — edit with care.\n"
    ]
    if domain is not None:
        parts.append(f"# HITL domain: {domain.value}\n")
    parts.append("\n")

    for i, fn in enumerate(hook_spec.functions):
        if i > 0:
            parts.append("\n\n")
        if fn.draft is None or not str(fn.draft).strip():
            raise HITLValidationError(
                f"Hook function {fn.name!r} has empty draft — cannot materialize."
            )
        parts.append(fn.draft.strip())

    return "".join(parts).rstrip() + "\n"


def _validate_module_ast(text: str, out_path: Path) -> None:
    from edvise.genai.mapping.identity_agent.hitl.resolver import HITLValidationError

    try:
        ast.parse(text)
    except SyntaxError as e:
        raise HITLValidationError(
            f"Assembled hook module failed ast.parse ({out_path}): {e}"
        ) from e


def _maybe_run_pyflakes(text: str, out_path: Path) -> None:
    try:
        from pyflakes.checker import Checker
    except ImportError:
        return
    try:
        tree = ast.parse(text)
        checker = Checker(tree, str(out_path))
        if checker.messages:
            logger.warning(
                "pyflakes reported issues in %s: %s",
                out_path,
                "; ".join(str(m) for m in checker.messages),
            )
    except Exception as e:
        logger.warning("pyflakes check skipped for %s: %s", out_path, e)


__all__ = [
    "materialize_hook_spec_to_file",
    "materialize_hook_specs_to_file",
    "merge_hook_specs",
]
