"""Unity Catalog volume path helpers for GenAI mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional


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


def bronze_volume_path_for_institution(
    institution_id: str,
    *,
    catalog: str = "",
) -> str:
    """``/Volumes/<catalog>/<institution_id>_bronze/bronze_volume``."""
    return _uc_volume_path(institution_id, catalog=catalog, tier="bronze")


def silver_genai_mapping_root(institution_id: str, *, catalog: str) -> str:
    """GenAI mapping run/active folders: ``…/silver_volume/genai_mapping``."""
    return f"{_uc_volume_path(institution_id, catalog=catalog, tier='silver')}/genai_mapping"


def resolve_genai_inputs_toml_path(
    institution_id: str,
    *,
    catalog: str,
    inputs_toml_path: Optional[str] = None,
) -> str:
    """
    Resolve IdentityAgent ``inputs.toml`` on UC volumes.

    Blank ``inputs_toml_path`` → ``…/bronze_volume/genai_mapping/inputs.toml``.
    Absolute paths are returned unchanged; relative paths join under ``genai_mapping/``.
    """
    base = (
        Path(bronze_volume_path_for_institution(institution_id, catalog=catalog))
        / "genai_mapping"
    )
    raw = (inputs_toml_path or "").strip()
    if not raw:
        return str(base / "inputs.toml")
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(base / raw.lstrip("/"))


def resolve_genai_data_path(bronze_volumes_path: Optional[str], file_path: str) -> str:
    """Join relative ``file_path`` to ``bronze_volumes_path``; absolute paths unchanged."""
    if not bronze_volumes_path or not str(bronze_volumes_path).strip():
        return file_path
    p = Path(file_path)
    if p.is_absolute():
        return file_path
    root = Path(bronze_volumes_path.rstrip("/"))
    return str(root / p)
