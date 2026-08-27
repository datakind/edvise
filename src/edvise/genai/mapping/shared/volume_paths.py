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


def genai_mapping_schema_volume_root(
    *, catalog: str, volume: str = "references"
) -> str:
    """
    Shared platform volume under the ``genai_mapping`` UC schema (not per-institution silver).

    Path: ``/Volumes/<catalog>/genai_mapping/<volume>``.
    """
    cat = _require_uc_catalog(catalog)
    vol = str(volume).strip()
    if not vol:
        raise ValueError("volume must be non-empty")
    return f"/Volumes/{cat}/genai_mapping/{vol}"


def genai_reference_root(reference_id: str, *, catalog: str) -> str:
    """``…/genai_mapping/references/<reference_id>`` on the shared references volume."""
    ref = str(reference_id).strip()
    if not ref:
        raise ValueError("reference_id must be non-empty")
    return f"{genai_mapping_schema_volume_root(catalog=catalog, volume='references')}/{ref}"


def genai_reference_current_root(reference_id: str, *, catalog: str) -> str:
    """Pinned few-shot artifacts: ``…/references/<reference_id>/current``."""
    return f"{genai_reference_root(reference_id, catalog=catalog)}/current"


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
