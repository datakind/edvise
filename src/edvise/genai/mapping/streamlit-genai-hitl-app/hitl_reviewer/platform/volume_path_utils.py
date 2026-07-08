"""Helpers for Unity Catalog silver volume paths used by GenAI mapping."""

from __future__ import annotations

import re
from typing import Literal


def _require_uc_catalog(catalog: str) -> str:
    cat = str(catalog).strip()
    if not cat:
        raise ValueError(
            "catalog (Databricks UC workspace catalog, e.g. job DB_workspace / --catalog) "
            "is required to resolve institution volume paths."
        )
    return cat


def _uc_volume_path(
    institution_id: str,
    *,
    catalog: str,
    tier: Literal["bronze", "silver"],
) -> str:
    inst = institution_id.strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    cat = _require_uc_catalog(catalog)
    return f"/Volumes/{cat}/{inst}_{tier}/{tier}_volume"


def silver_genai_mapping_root(institution_id: str, *, catalog: str) -> str:
    """GenAI mapping run/active folders: ``…/silver_volume/genai_mapping``."""
    return f"{_uc_volume_path(institution_id, catalog=catalog, tier='silver')}/genai_mapping"


# /Volumes/<catalog>/<institution_id>_silver/silver_volume/...
_INST_SILVER = re.compile(
    r"^/Volumes/[^/]+/(?P<inst>[^/]+)_silver/silver_volume/",
)


def institution_id_from_silver_volume_path(volume_path: str) -> str | None:
    """
    Parse ``institution_id`` from a path under ``…/<institution_id>_silver/silver_volume/``.

    Returns None when the pattern does not match.
    """
    p = (volume_path or "").strip()
    m = _INST_SILVER.match(p)
    if not m:
        return None
    return m.group("inst").strip() or None
