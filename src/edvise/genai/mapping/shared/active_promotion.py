"""
Promote GenAI mapping onboard artifacts into ``genai_mapping/active/`` (execute-mode layout).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Protocol

LOGGER = logging.getLogger(__name__)

_IDENTITY_OPTIONAL_ACTIVE: tuple[tuple[str, str], ...] = (
    ("identity_grain_output.json", "grain_output.json"),
    ("identity_term_output.json", "term_output.json"),
    ("term_hooks.py", "term_hooks.py"),
    ("grain_hooks.py", "grain_hooks.py"),
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


def promote_genai_mapping_to_active(paths: _ActivePromotionPaths) -> None:
    """
    After a successful SMA onboard ``gate_2``, copy canonical artifacts from the run tree into
    ``paths.active_root`` so ``mode=execute`` can load them.

    Required sources: IA ``enriched_schema_contract.json``, SMA ``manifest_map.json`` and
    ``transformation_map.json``. Optional: ``transform_hooks.py`` if present; identity-agent
    outputs/hooks when present.
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
        LOGGER.info("Promoted %s -> %s", paths.transform_hooks, paths.active_transform_hooks)

    for src_name, dst_name in _IDENTITY_OPTIONAL_ACTIVE:
        src, dst = ia_root / src_name, paths.active_root / dst_name
        if src.is_file():
            shutil.copy2(src, dst)
            LOGGER.info("Promoted %s -> %s", src, dst)
