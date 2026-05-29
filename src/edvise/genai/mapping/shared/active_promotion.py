"""
Promote GenAI mapping onboard artifacts into ``genai_mapping/active/`` (execute-mode layout).

After a successful promotion, writes :data:`GENAI_ACTIVE_REGISTRY_BASENAME` (model-registry style
metadata: source ``onboard_run_id``, ``institution_id``, ``promoted_at``, etc.).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)

GENAI_ACTIVE_REGISTRY_BASENAME: str = "genai_active_registry.json"
"""JSON sidecar under ``active/`` recording which onboard run produced the promoted artifacts."""

_ACTIVE_REGISTRY_SCHEMA_VERSION: int = 1

_IDENTITY_OPTIONAL_ACTIVE: tuple[tuple[str, str], ...] = (
    ("identity_grain_output.json", "grain_output.json"),
    ("identity_term_output.json", "term_output.json"),
    # Legacy flat hook filenames (pre-identity_hooks/ layout); canonical modules live under
    # identity_hooks/<institution_id>/ and are promoted via _promote_identity_hooks_subtree.
    ("term_hooks.py", "term_hooks.py"),
    ("grain_hooks.py", "grain_hooks.py"),
)

# SMA gate-2 grain resolution sidecars (same basename under ``schema_mapping_agent/`` and ``active/``).
_SMA_GRAIN_RESOLUTION_FILENAMES: tuple[str, ...] = (
    "sma_grain_resolution_cohort.json",
    "sma_grain_resolution_course.json",
)


class _ActivePromotionPaths(Protocol):
    """Subset of :class:`~edvise.genai.mapping.scripts.edvise_genai_sma.SMAPaths` used for promotion."""

    active_root: Path
    active_enriched_schema_contract: Path
    active_manifest_map: Path
    active_transformation_map: Path
    active_transform_hooks: Path
    ia_enriched_schema_contract: Path
    manifest_map: Path
    transformation_map: Path
    transform_hooks: Path


def read_genai_active_registry(active_root: str | Path) -> dict[str, Any] | None:
    """
    Load ``genai_active_registry.json`` from ``active_root`` if present.

    Returns None when the file is missing (e.g. legacy promotions before this metadata existed).
    """
    p = Path(active_root) / GENAI_ACTIVE_REGISTRY_BASENAME
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {p}")
    return data


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_genai_active_registry(
    active_root: Path,
    *,
    institution_id: str,
    onboard_run_id: str,
    pipeline_version: str | None,
    uc_catalog: str | None,
) -> None:
    inst = str(institution_id).strip()
    rid = str(onboard_run_id).strip()
    if not inst or not rid:
        raise ValueError("institution_id and onboard_run_id must be non-empty")
    payload: dict[str, Any] = {
        "schema_version": _ACTIVE_REGISTRY_SCHEMA_VERSION,
        "onboard_run_id": rid,
        "institution_id": inst,
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if pipeline_version and str(pipeline_version).strip():
        payload["pipeline_version"] = str(pipeline_version).strip()
    if uc_catalog and str(uc_catalog).strip():
        payload["uc_catalog"] = str(uc_catalog).strip()
    dest = active_root / GENAI_ACTIVE_REGISTRY_BASENAME
    _atomic_write_json(dest, payload)
    LOGGER.info("Wrote %s (onboard_run_id=%r)", dest, rid)


def _promote_identity_hooks_subtree(*, ia_root: Path, active_root: Path) -> None:
    """
    Copy materialized IA hook modules so ``hook_spec.file`` paths such as
    ``identity_hooks/<institution_id>/dedup_hooks.py`` resolve under ``active_root``.
    """
    src = ia_root / "identity_hooks"
    if not src.is_dir():
        return
    dst = active_root / "identity_hooks"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    LOGGER.info("Promoted identity_hooks tree %s -> %s", src, dst)


def promote_genai_mapping_to_active(
    paths: _ActivePromotionPaths,
    *,
    institution_id: str,
    onboard_run_id: str,
    pipeline_version: str | None = None,
    uc_catalog: str | None = None,
) -> None:
    """
    After a successful SMA onboard ``gate_2``, copy canonical artifacts from the run tree into
    ``paths.active_root`` so ``mode=execute`` can load them.

    Required sources: IA ``enriched_schema_contract.json``, SMA ``manifest_map.json`` and
    ``transformation_map.json``. Optional: ``transform_hooks.py`` if present; identity-agent
    outputs when present; ``identity_hooks/`` subtree when materialized hook modules exist
    (matches :func:`~edvise.genai.mapping.shared.hitl.hook_spec.paths.default_hook_module_relpath`).
    Optional SMA grain resolution JSON next to the onboard ``manifest_map.json``
    (``sma_grain_resolution_cohort.json`` / ``sma_grain_resolution_course.json``) — copied into
    ``active/`` when present so ``mode=execute`` can load them (see
    :func:`~edvise.genai.mapping.schema_mapping_agent.grain_resolution.runner.execute_transformation_map_for_sma_run`).

    Writes :data:`GENAI_ACTIVE_REGISTRY_BASENAME` last (atomic) so ``active/`` records the source
    ``onboard_run_id`` for this promotion.
    """
    paths.active_root.mkdir(parents=True, exist_ok=True)
    ia_root = paths.ia_enriched_schema_contract.parent

    required: list[tuple[Path, Path]] = [
        (paths.ia_enriched_schema_contract, paths.active_enriched_schema_contract),
        (paths.manifest_map, paths.active_manifest_map),
        (paths.transformation_map, paths.active_transformation_map),
    ]
    for src, dst in required:
        if not src.is_file():
            raise FileNotFoundError(f"Cannot promote missing artifact: {src}")
        shutil.copy2(src, dst)
        LOGGER.info("Promoted %s -> %s", src, dst)

    if paths.transform_hooks.is_file():
        shutil.copy2(paths.transform_hooks, paths.active_transform_hooks)
        LOGGER.info(
            "Promoted %s -> %s", paths.transform_hooks, paths.active_transform_hooks
        )

    _promote_identity_hooks_subtree(ia_root=ia_root, active_root=paths.active_root)

    for src_name, dst_name in _IDENTITY_OPTIONAL_ACTIVE:
        src, dst = ia_root / src_name, paths.active_root / dst_name
        if src.is_file():
            shutil.copy2(src, dst)
            LOGGER.info("Promoted %s -> %s", src, dst)

    sma_run_root = paths.manifest_map.parent
    for fname in _SMA_GRAIN_RESOLUTION_FILENAMES:
        src = sma_run_root / fname
        if src.is_file():
            dst = paths.active_root / fname
            shutil.copy2(src, dst)
            LOGGER.info("Promoted %s -> %s", src, dst)

    _write_genai_active_registry(
        paths.active_root,
        institution_id=institution_id,
        onboard_run_id=onboard_run_id,
        pipeline_version=pipeline_version,
        uc_catalog=uc_catalog,
    )
