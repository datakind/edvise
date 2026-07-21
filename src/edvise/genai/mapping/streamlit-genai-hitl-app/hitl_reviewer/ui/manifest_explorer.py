"""
Manifest Explorer: one consolidated, filterable table joining the Step 2a field mapping
manifest (``manifest_map.json``) with source-column stats (from
``enriched_schema_contract.json``) and HITL review status (``cohort_hitl_manifest.json`` /
``course_hitl_manifest.json``).

Today a reviewer has to open each of those files separately (see
``hitl_reviewer.ui.run_artifacts_browser``) and cross-reference target fields and source
columns by eye. This module assembles them into a single ``pandas.DataFrame`` — one row per
target field — so the app can render one table instead of several raw-JSON expanders.

Kept free of ``edvise`` package imports (same constraint as
:mod:`hitl_reviewer.ui.sma.enriched_schema_contract`) so the Databricks app bundle does not
need the full ``edvise`` install.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hitl_reviewer.platform.unity_volume_files import read_unity_file_text
from hitl_reviewer.platform.volume_path_utils import silver_genai_mapping_root
from hitl_reviewer.ui.sma.enriched_schema_contract import (
    extract_column_panel_fields,
    load_json_object_from_text,
    visualize_value_whitespace,
)

# Columns always present on the returned table, even when a source file is missing/unreadable
# or the manifest is empty — callers can rely on these existing without checking first.
_HITL_STATUS_COLUMNS: tuple[str, ...] = (
    "flagged_for_hitl",
    "hitl_failure_mode",
    "hitl_resolved",
    "hitl_question",
)
_COLUMN_STATS_COLUMNS: tuple[str, ...] = (
    "dtype",
    "null_percentage",
    "null_count",
    "unique_count",
    "sample_values",
    "has_padded_values",
)


def genai_mapping_root_uc(institution_id: str, catalog: str) -> str:
    return silver_genai_mapping_root(
        str(institution_id).strip(), catalog=str(catalog).strip()
    )


def resolve_explorer_paths(
    institution_id: str,
    catalog: str,
    *,
    onboard_run_id: str | None,
) -> dict[str, str]:
    """
    Absolute Files-API paths for the artifacts this page joins together.

    ``onboard_run_id`` set → paths under ``runs/onboard/{run_id}/…`` (manifest may be mid-HITL;
    the two HITL manifest paths may or may not exist depending on run state).
    ``onboard_run_id`` None → promoted ``active/…`` paths (post-HITL; HITL manifest paths are
    returned empty since Active has no per-field HITL items).
    """
    root = genai_mapping_root_uc(institution_id, catalog)
    rid = str(onboard_run_id or "").strip()
    if rid:
        base = Path(root) / "runs" / "onboard" / rid
        sma = base / "schema_mapping_agent"
        ia = base / "identity_agent"
        return {
            "manifest_map": str(sma / "manifest_map.json"),
            "enriched_schema_contract": str(ia / "enriched_schema_contract.json"),
            "cohort_hitl_manifest": str(sma / "cohort_hitl_manifest.json"),
            "course_hitl_manifest": str(sma / "course_hitl_manifest.json"),
        }
    active = Path(root) / "active"
    return {
        "manifest_map": str(active / "manifest_map.json"),
        "enriched_schema_contract": str(active / "enriched_schema_contract.json"),
        "cohort_hitl_manifest": "",
        "course_hitl_manifest": "",
    }


def load_json_object_or_none(abs_path: str) -> dict[str, Any] | None:
    """``None`` on any read/parse failure (missing file, permissions, not JSON, not an object)."""
    p = (abs_path or "").strip()
    if not p:
        return None
    try:
        raw = read_unity_file_text(p)
    except Exception:  # noqa: BLE001
        return None
    try:
        return load_json_object_from_text(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _join_summary(join: Any) -> str:
    if not isinstance(join, dict) or not join:
        return ""
    bt = join.get("base_table")
    lt = join.get("lookup_table")
    jk = join.get("join_keys")
    jk_s = ", ".join(str(k) for k in jk) if isinstance(jk, list) else str(jk)
    return f"{bt} ← {lt} on [{jk_s}]"


def _row_selection_summary(rs: Any) -> str:
    if not isinstance(rs, dict) or not rs:
        return ""
    parts = [str(rs.get("strategy") or "")]
    if rs.get("order_by") is not None:
        parts.append(f"order_by={rs['order_by']}")
    if rs.get("condition_col") is not None:
        parts.append(f"condition_col={rs['condition_col']}")
    if rs.get("n") is not None:
        parts.append(f"n={rs['n']}")
    filt = rs.get("filter")
    if isinstance(filt, dict) and filt.get("column"):
        parts.append(
            f"filter: {filt.get('column')} {filt.get('operator')} {filt.get('value')!r}"
        )
    return " · ".join(p for p in parts if p)


def flatten_manifest_envelope(envelope: dict[str, Any] | None) -> pd.DataFrame:
    """One row per ``FieldMappingRecord`` across the cohort and course manifests."""
    rows: list[dict[str, Any]] = []
    manifests = (envelope or {}).get("manifests") or {}
    if isinstance(manifests, dict):
        for entity_type in ("cohort", "course"):
            manifest = manifests.get(entity_type)
            if not isinstance(manifest, dict):
                continue
            mappings = manifest.get("mappings")
            if not isinstance(mappings, list):
                continue
            for rec in mappings:
                if not isinstance(rec, dict):
                    continue
                rows.append(
                    {
                        "entity_type": entity_type,
                        "target_field": rec.get("target_field"),
                        "source_table": rec.get("source_table"),
                        "source_column": rec.get("source_column"),
                        "join": _join_summary(rec.get("join")),
                        "row_selection": _row_selection_summary(rec.get("row_selection")),
                        "confidence": rec.get("confidence"),
                        "review_status": rec.get("review_status") or "",
                        "rationale": rec.get("rationale") or "",
                        "validation_notes": rec.get("validation_notes") or "",
                        "reviewer_notes": rec.get("reviewer_notes") or "",
                        "_join_raw": rec.get("join"),
                        "_row_selection_raw": rec.get("row_selection"),
                    }
                )
    cols = [
        "entity_type",
        "target_field",
        "source_table",
        "source_column",
        "join",
        "row_selection",
        "confidence",
        "review_status",
        "rationale",
        "validation_notes",
        "reviewer_notes",
        "_join_raw",
        "_row_selection_raw",
    ]
    return pd.DataFrame(rows, columns=cols)


def _hitl_lookup(hitl_json: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``target_field -> {failure_mode, choice, hitl_question}`` from a cohort/course HITL manifest."""
    out: dict[str, dict[str, Any]] = {}
    items = (hitl_json or {}).get("items")
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        tf = it.get("target_field")
        if tf is None:
            continue
        out[str(tf)] = {
            "failure_mode": it.get("failure_mode") or "",
            "hitl_choice": it.get("choice"),
            "hitl_question": it.get("hitl_question") or "",
        }
    return out


def attach_hitl_status(
    df: pd.DataFrame,
    *,
    cohort_hitl: dict[str, Any] | None,
    course_hitl: dict[str, Any] | None,
) -> pd.DataFrame:
    """Adds ``flagged_for_hitl`` / ``hitl_failure_mode`` / ``hitl_resolved`` / ``hitl_question``."""
    if df.empty:
        for c in _HITL_STATUS_COLUMNS:
            df[c] = pd.Series(dtype="object")
        return df
    lookups = {"cohort": _hitl_lookup(cohort_hitl), "course": _hitl_lookup(course_hitl)}

    def _row(r: pd.Series) -> pd.Series:
        lk = lookups.get(str(r["entity_type"]), {})
        hit = lk.get(str(r["target_field"]))
        if hit is None:
            return pd.Series(
                {
                    "flagged_for_hitl": False,
                    "hitl_failure_mode": "",
                    "hitl_resolved": None,
                    "hitl_question": "",
                }
            )
        return pd.Series(
            {
                "flagged_for_hitl": True,
                "hitl_failure_mode": hit.get("failure_mode") or "",
                "hitl_resolved": hit.get("hitl_choice") is not None,
                "hitl_question": hit.get("hitl_question") or "",
            }
        )

    extra = df.apply(_row, axis=1)
    return pd.concat([df, extra], axis=1)


def attach_column_stats(df: pd.DataFrame, contract: dict[str, Any] | None) -> pd.DataFrame:
    """
    Joins per-row source-column stats (dtype, null rate, unique count, sample values) from the
    enriched schema contract. Reuses :func:`extract_column_panel_fields` — the same lookup the
    per-item SMA HITL panel uses for the single currently-flagged item — applied here to every
    row so the whole manifest can be reviewed against column quality at once.
    """
    if df.empty or not contract:
        for c in _COLUMN_STATS_COLUMNS:
            df[c] = pd.Series(dtype="object")
        return df

    def _row(r: pd.Series) -> pd.Series:
        table = r.get("source_table")
        column = r.get("source_column")
        if not table or not column:
            return pd.Series({c: None for c in _COLUMN_STATS_COLUMNS})
        panel = extract_column_panel_fields(
            contract, dataset_name=str(table), source_column=str(column)
        )
        if panel is None:
            return pd.Series({c: None for c in _COLUMN_STATS_COLUMNS})
        chips = panel.get("chip_values") or []
        preview_cap = 12
        displayed: list[str] = []
        any_padded = False
        for v in chips[:preview_cap]:
            disp, had_padding = visualize_value_whitespace(v)
            any_padded = any_padded or had_padding
            displayed.append(disp)
        preview = ", ".join(displayed)
        remaining = len(chips) - preview_cap
        if remaining > 0:
            preview += f", … (+{remaining} more — see detail panel)"
        return pd.Series(
            {
                "dtype": panel.get("dtype"),
                "null_percentage": panel.get("null_percentage"),
                "null_count": panel.get("null_count"),
                "unique_count": panel.get("unique_count"),
                "sample_values": preview,
                "has_padded_values": any_padded,
            }
        )

    extra = df.apply(_row, axis=1)
    return pd.concat([df, extra], axis=1)


def build_manifest_explorer_table(
    *,
    manifest_map_path: str,
    enriched_schema_contract_path: str,
    cohort_hitl_manifest_path: str,
    course_hitl_manifest_path: str,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """
    Returns ``(table, availability)``. ``availability`` reports which source files were
    readable so the page can tell the reviewer what's missing rather than silently rendering
    blank columns.
    """
    envelope = load_json_object_or_none(manifest_map_path)
    contract = load_json_object_or_none(enriched_schema_contract_path)
    cohort_hitl = load_json_object_or_none(cohort_hitl_manifest_path)
    course_hitl = load_json_object_or_none(course_hitl_manifest_path)

    df = flatten_manifest_envelope(envelope)
    df = attach_hitl_status(df, cohort_hitl=cohort_hitl, course_hitl=course_hitl)
    df = attach_column_stats(df, contract)

    availability = {
        "manifest_map": envelope is not None,
        "enriched_schema_contract": contract is not None,
        "cohort_hitl_manifest": bool(cohort_hitl_manifest_path) and cohort_hitl is not None,
        "course_hitl_manifest": bool(course_hitl_manifest_path) and course_hitl is not None,
    }
    return df, availability


def full_column_sample_values(
    contract: dict[str, Any] | None,
    *,
    source_table: str | None,
    source_column: str | None,
    limit: int = 80,
) -> tuple[list[str], str]:
    """Untruncated (up to ``limit``) chip values + mode ('unique'|'sample') for a detail panel."""
    if not contract or not source_table or not source_column:
        return [], "sample"
    panel = extract_column_panel_fields(
        contract, dataset_name=str(source_table), source_column=str(source_column)
    )
    if panel is None:
        return [], "sample"
    chips = panel.get("chip_values") or []
    return [str(c) for c in chips[:limit]], str(panel.get("chip_mode") or "sample")
