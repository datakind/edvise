"""Resolve GenAI ``pipeline_version`` from ``genai_active_registry.json`` for ES materialize."""

from __future__ import annotations

import logging
from pathlib import Path

from edvise.genai.mapping.shared.active_promotion import (
    GENAI_ACTIVE_REGISTRY_BASENAME,
    read_genai_active_registry,
)

LOGGER = logging.getLogger(__name__)


def silver_volume_root(db_workspace: str, databricks_institution_name: str) -> Path:
    """``/Volumes/{catalog}/{institution}_silver/silver_volume``."""
    catalog = db_workspace.strip()
    inst = databricks_institution_name.strip()
    if not catalog or not inst:
        raise ValueError(
            "db_workspace and databricks_institution_name are required "
            "to resolve the GenAI active registry path."
        )
    return Path(f"/Volumes/{catalog}/{inst}_silver/silver_volume")


def genai_active_root(db_workspace: str, databricks_institution_name: str) -> Path:
    """``…/silver_volume/genai_mapping/active``."""
    return (
        silver_volume_root(db_workspace, databricks_institution_name)
        / ("genai_mapping")
        / "active"
    )


def resolve_genai_pipeline_version_from_registry(
    db_workspace: str,
    databricks_institution_name: str,
    *,
    logger: logging.Logger = LOGGER,
) -> str:
    """
    Read ``pipeline_version`` from the institution's GenAI active registry.

    Raises ``FileNotFoundError`` / ``ValueError`` when the registry or field is missing.
    """
    active_root = genai_active_root(db_workspace, databricks_institution_name)
    registry_path = active_root / GENAI_ACTIVE_REGISTRY_BASENAME
    payload = read_genai_active_registry(active_root)
    if payload is None:
        raise FileNotFoundError(
            f"GenAI active registry not found at {registry_path}. "
            "GenAI schools require genai_mapping/active/genai_active_registry.json "
            "with pipeline_version from promotion."
        )
    raw = payload.get("pipeline_version")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"GenAI active registry at {registry_path} has no non-empty "
            f"'pipeline_version' (got {raw!r})."
        )
    version = raw.strip()
    logger.info(
        "Resolved GenAI pipeline_version=%s from %s",
        version,
        registry_path,
    )
    return version
