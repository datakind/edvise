"""
SMA HITL: resolve ``identity_agent/enriched_schema_contract.json`` and extract
per-column training stats for the reviewer panel.

Kept free of ``edvise`` imports for the Databricks app bundle.
"""

from __future__ import annotations

import json
import re
from typing import Any


def silver_relative_path(absolute_volume_path: str) -> str | None:
    """Strip ``/Volumes/{catalog}/{inst}_silver/`` → ``silver_volume/...``."""
    p = (absolute_volume_path or "").strip()
    m = re.match(r"^/Volumes/[^/]+/[^/]+_silver/(.+)$", p)
    return m.group(1) if m else None


def enriched_schema_contract_path_from_manifest(
    manifest_absolute_path: str, onboard_run_id: str
) -> str | None:
    """
    ``…/runs/onboard/{onboard_run_id}/schema_mapping_agent/…``
    → ``…/runs/onboard/{onboard_run_id}/identity_agent/enriched_schema_contract.json``
    """
    p = (manifest_absolute_path or "").strip()
    rid = (onboard_run_id or "").strip()
    if not p.startswith("/Volumes/") or not rid:
        return None
    needle = f"/runs/onboard/{rid}/"
    pos = p.find(needle)
    if pos < 0:
        return None
    base = p[: pos + len(needle)]
    return f"{base}identity_agent/enriched_schema_contract.json"


def _as_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if x is None:
            continue
        out.append(str(x))
    return out


def extract_column_panel_fields(
    contract: dict[str, Any],
    *,
    dataset_name: str,
    source_column: str | None,
) -> dict[str, Any] | None:
    """
    Returns a dict with keys:
      original_name, dtype, null_percentage, null_count, unique_count,
      chip_values, chip_mode ('unique'|'sample'), inst_null_tokens, dataset_null_tokens
    or None if not found.
    """
    if not source_column:
        return None
    col = str(source_column).strip()
    if not col:
        return None
    ds = (contract.get("datasets") or {}).get(dataset_name)
    if not isinstance(ds, dict):
        return None
    dtypes = ds.get("dtypes") if isinstance(ds.get("dtypes"), dict) else {}
    dtype = str(dtypes.get(col, "") or "").strip() or "—"
    training = ds.get("training") if isinstance(ds.get("training"), dict) else {}
    details = training.get("column_details")
    if not isinstance(details, list):
        return None
    row: dict[str, Any] | None = None
    for entry in details:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("normalized_name", "")).strip() == col:
            row = entry
            break
    if row is None:
        return None
    inst_tokens = _as_str_list(contract.get("null_tokens"))
    ds_tokens = _as_str_list(ds.get("null_tokens"))
    unique_values = row.get("unique_values")
    sample_values = _as_str_list(row.get("sample_values"))
    chips: list[str]
    chip_mode: str
    if isinstance(unique_values, list) and len(unique_values) <= 20:
        chips = [str(x) for x in unique_values if x is not None]
        chip_mode = "unique"
    else:
        chips = sample_values
        chip_mode = "sample"
    return {
        "original_name": str(row.get("original_name", "") or "").strip() or "—",
        "dtype": dtype,
        "null_percentage": row.get("null_percentage"),
        "null_count": row.get("null_count"),
        "unique_count": row.get("unique_count"),
        "chip_values": chips,
        "chip_mode": chip_mode,
        "inst_null_tokens": inst_tokens,
        "dataset_null_tokens": ds_tokens,
    }


def load_json_object_from_text(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("Expected JSON object")
    return data
