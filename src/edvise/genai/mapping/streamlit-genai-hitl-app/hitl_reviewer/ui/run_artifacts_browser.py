"""
Browse GenAI mapping artifacts on silver: onboard run outputs and Active-folder promotion.

Uses the same path layout as ``edvise_genai_ia`` / ``edvise_genai_sma`` (``resolve_run_paths``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import streamlit as st

from edvise.configs import genai as genai_cfg
from hitl_reviewer.platform.unity_volume_files import read_unity_file_text

_PREVIEW_CHAR_CAP = 120_000


def _ia_hook_module_relpath(institution_id: str, *, grain: bool) -> str:
    """
    Canonical relative path under the IA onboard run root for materialized hook modules.

    Matches :func:`~edvise.genai.mapping.shared.hitl.hook_spec.paths.default_hook_module_relpath`
    for ``identity_grain`` / ``identity_term`` only. Kept local so this module does not import
    ``hook_spec`` (older ``edvise`` installs used by the Streamlit app may omit that package).
    """
    basename = "dedup_hooks.py" if grain else "term_hooks.py"
    return f"identity_hooks/{institution_id}/{basename}"


def genai_mapping_root_uc(institution_id: str, catalog: str) -> str:
    return genai_cfg.silver_genai_mapping_root(
        str(institution_id).strip(), catalog=str(catalog).strip()
    )


def known_onboard_run_artifact_paths(
    institution_id: str, catalog: str, onboard_run_id: str
) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Return ``(genai_mapping_root, [(section, label, absolute_path), ...])`` for one onboard run.

    Candidate paths for this run; the UI only shows entries that are successfully read from the volume.
    """
    inst = str(institution_id).strip()
    cat = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    root = genai_mapping_root_uc(inst, cat)
    base = Path(root) / "runs" / "onboard" / rid
    ia = base / "identity_agent"
    sma = base / "schema_mapping_agent"
    items: list[tuple[str, str, str]] = []

    def add(section: str, label: str, p: Path) -> None:
        items.append((section, label, str(p)))

    for fn, label in (
        ("enriched_schema_contract.json", "Enriched schema contract"),
        ("identity_grain_output.json", "IA grain output"),
        ("identity_grain_hitl.json", "IA grain HITL"),
        ("identity_term_output.json", "IA term output"),
        ("identity_term_hitl.json", "IA term HITL"),
        ("identity_grain_hook_preview.json", "IA grain hook preview"),
        ("identity_term_hook_preview.json", "IA term hook preview"),
        ("profiling_output.json", "IA profiling output"),
        ("run_log.json", "IA run_log.json"),
    ):
        add("Identity agent (onboard run)", label, ia / fn)

    # Canonical materialized modules (same layout as ``default_hook_module_relpath`` for IA).
    for grain, label in (
        (True, "identity_hooks/…/dedup_hooks.py"),
        (False, "identity_hooks/…/term_hooks.py"),
    ):
        rel = _ia_hook_module_relpath(inst, grain=grain)
        add("Identity hooks (onboard run)", label, ia / rel)

    for fn, label in (
        ("manifest_map.json", "Manifest map"),
        ("transformation_map.json", "Transformation map"),
        ("mapping_validation_manifest.json", "Mapping validation manifest"),
        ("cohort_hitl_manifest.json", "Cohort manifest HITL"),
        ("course_hitl_manifest.json", "Course manifest HITL"),
        ("cohort_transformation_review.json", "Cohort transformation review"),
        ("course_transformation_review.json", "Course transformation review"),
        ("cohort_transformation_hook_preview.json", "Cohort hook preview"),
        ("course_transformation_hook_preview.json", "Course hook preview"),
        ("cohort_transformation_hook_hitl.json", "Cohort transformation hook HITL"),
        ("course_transformation_hook_hitl.json", "Course transformation hook HITL"),
        ("transform_hooks.py", "transform_hooks.py"),
        ("run_log.json", "SMA run_log.json"),
        ("repair_log.json", "SMA repair_log.json"),
        ("pandera_validation_errors.json", "Pandera validation errors"),
    ):
        add("Schema mapping agent (onboard run)", label, sma / fn)

    add(
        "SMA outputs (onboard run)",
        "Execute-mode parquet folder (if present)",
        base / "pipeline_input",
    )
    return root, items


def known_active_artifact_paths(
    institution_id: str, catalog: str
) -> list[tuple[str, str, str]]:
    """``[(section, label, absolute_path), ...]`` under ``genai_mapping/active/`` on the volume."""
    inst = str(institution_id).strip()
    cat = str(catalog).strip()
    active = Path(genai_mapping_root_uc(inst, cat)) / "active"
    items: list[tuple[str, str, str]] = []

    def add(label: str, p: Path) -> None:
        items.append(("Active", label, str(p)))

    for fn, label in (
        ("genai_active_registry.json", "Promotion registry"),
        ("enriched_schema_contract.json", "Enriched schema contract"),
        ("manifest_map.json", "Manifest map"),
        ("transformation_map.json", "Transformation map"),
        ("transform_hooks.py", "transform_hooks.py"),
        ("grain_output.json", "Grain output"),
        ("term_output.json", "Term output"),
    ):
        add(label, active / fn)

    for grain, label in (
        (True, "identity_hooks/…/dedup_hooks.py"),
        (False, "identity_hooks/…/term_hooks.py"),
    ):
        rel = _ia_hook_module_relpath(inst, grain=grain)
        add(label, active / rel)

    return items


def _read_uc_file_if_accessible(abs_path: str) -> str | None:
    """
    Return file text if the Files API read succeeds, else ``None`` (missing, denied, wrong path, etc.).
    """
    p = (abs_path or "").strip()
    if not p.startswith("/Volumes/"):
        return None
    try:
        return read_unity_file_text(p)
    except Exception:  # noqa: BLE001
        return None


def _preview_payload(text: str) -> tuple[str, Any | None]:
    """Returns (preview_string, parsed_json_or_none for ``st.json``)."""
    if len(text) > _PREVIEW_CHAR_CAP:
        preview = (
            text[:_PREVIEW_CHAR_CAP]
            + f"\n\n… truncated preview ({_PREVIEW_CHAR_CAP} chars); use Download for full file."
        )
    else:
        preview = text
    if len(text) > 5_000_000:
        return preview, None
    try:
        return preview, json.loads(text)
    except json.JSONDecodeError:
        return preview, None


def _render_directory_data_expander(section: str, label: str, abs_path: str) -> None:
    title = f"{section} — {label}"
    with st.expander(title, expanded=False):
        st.code(abs_path, language="text")
        st.info(
            "Directory for execute-mode parquet outputs when present — open this path in "
            "Databricks / Unity Catalog to list files."
        )


def _render_file_expander(section: str, label: str, abs_path: str, body: str) -> None:
    title = f"{section} — {label}"
    with st.expander(title, expanded=False):
        st.code(abs_path, language="text")
        if not body.strip():
            st.caption("(empty file)")
        preview, as_json = _preview_payload(body)
        dl_key = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()[:24]
        st.download_button(
            label="Download full file",
            data=body.encode("utf-8"),
            file_name=Path(abs_path).name,
            key=f"dl-{dl_key}",
        )
        if as_json is not None:
            st.json(as_json)
        else:
            st.text(preview if body.strip() else "")


def render_artifact_sections(
    *,
    title: str,
    paths: list[tuple[str, str, str]],
) -> None:
    st.subheader(title)
    if not paths:
        st.caption("No paths to show.")
        return
    shown = 0
    for section, label, abs_path in paths:
        if Path(abs_path).name == "data":
            _render_directory_data_expander(section, label, abs_path)
            shown += 1
            continue
        body = _read_uc_file_if_accessible(abs_path)
        if body is None:
            continue
        _render_file_expander(section, label, abs_path, body)
        shown += 1
    if shown == 0:
        st.caption(
            "No files could be read at the expected paths (missing, permissions, or not a volume path)."
        )
