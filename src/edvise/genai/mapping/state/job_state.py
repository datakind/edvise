"""
Best-effort pipeline state updates for Databricks job entrypoints (IA / SMA).

Failures are logged and do not block the job (same spirit as
:func:`~edvise.genai.mapping.shared.pipeline_artifacts.merge_genai_pipeline_artifact_rows`).

UC HITL polling helpers (:func:`wait_for_ia_gate_1_hitl`, :func:`wait_for_sma_gate_1_hitl`) are
blocking and raise on timeout or rejection. Timeouts persist ``timed_out`` on ``pipeline_runs`` /
``pipeline_phases`` (resumable); other failures may use :func:`mark_pipeline_failed`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from edvise.genai.mapping.state import pipeline_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    poll_uc_hitl_until_approved_or_timeout,
)

LOGGER = logging.getLogger(__name__)

PHASE_IA_START: str = "ia_start"
PHASE_IA_GATE_1: str = "ia_gate_1"
PHASE_SMA_START: str = "sma_start"
PHASE_SMA_GATE_1: str = "sma_gate_1"


def _state_safe(label: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — intentional non-fatal
        LOGGER.warning("Pipeline state [%s] skipped: %s", label, e)


def mark_pipeline_failed(catalog: str, institution_id: str, onboard_run_id: str) -> None:
    _state_safe(
        "update_pipeline_run_status(failed)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "failed",
    )


def after_ia_onboard_start(
    catalog: str, institution_id: str, onboard_run_id: str, *, grain_path: Path, term_path: Path
) -> None:
    g = grain_path.as_posix()
    t = term_path.as_posix()
    _state_safe(
        "ia_start -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_IA_START,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (IA)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (ia_gate_1 targets)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1,
        [
            {"artifact_type": "grain", "artifact_path": g},
            {"artifact_type": "term", "artifact_path": t},
        ],
    )


def on_ia_onboard_begin(
    catalog: str,
    onboard_run_id: str,
    *,
    resume_from: str,
    institution_id: str | None = None,
    input_file_paths_json: str | None = None,
) -> None:
    if resume_from == "start":
        _state_safe(
            "ia_start running",
            pipeline_state.log_phase_transition,
            catalog,
            onboard_run_id,
            PHASE_IA_START,
            "running",
        )
    else:
        _state_safe(
            "ia_gate_1 running",
            pipeline_state.log_phase_transition,
            catalog,
            onboard_run_id,
            PHASE_IA_GATE_1,
            "running",
        )
    if (
        resume_from == "start"
        and (institution_id or "").strip()
        and (input_file_paths_json or "").strip()
    ):
        _state_safe(
            "pipeline_runs input_file_paths (IA onboard begin)",
            pipeline_state.update_onboard_pipeline_run_input_file_paths,
            catalog,
            str(institution_id).strip(),
            onboard_run_id,
            str(input_file_paths_json).strip(),
        )


def wait_for_ia_gate_1_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """
    Block until every ``hitl_reviews`` row for ``ia_gate_1`` is ``approved`` in Unity Catalog.

    Used at the beginning of IA onboard ``resume_from=gate_1`` before local JSON HITL gates.
    """
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_IA_GATE_1,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def wait_for_sma_gate_1_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """
    Block until every ``hitl_reviews`` row for ``sma_gate_1`` is ``approved`` in Unity Catalog.

    Used at the beginning of SMA onboard ``resume_from=gate_2`` (second step) before resolving
    manifest HITL JSON on disk.
    """
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_SMA_GATE_1,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def after_ia_onboard_gate_1_success(catalog: str, institution_id: str, onboard_run_id: str) -> None:
    _state_safe(
        "ia_gate_1 complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> running (post-IA, pre-SMA)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "running",
    )


def ensure_ia_run_row(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    create_run: bool,
    db_run_id: str | None = None,
    input_file_paths_json: str | None = None,
) -> None:
    if not create_run:
        return
    _state_safe(
        "create_pipeline_run",
        pipeline_state.create_pipeline_run,
        catalog,
        institution_id,
        onboard_run_id,
        db_run_id,
        input_file_paths_json,
    )


# --- SMA ------------------------------------------------------------------


def on_sma_onboard_begin(
    catalog: str,
    onboard_run_id: str,
    *,
    resume_from: str,
    institution_id: str | None = None,
    input_file_paths_json: str | None = None,
) -> None:
    if resume_from == "start":
        _state_safe(
            "sma_start running",
            pipeline_state.log_phase_transition,
            catalog,
            onboard_run_id,
            PHASE_SMA_START,
            "running",
        )
    else:
        _state_safe(
            "sma_gate_1 running",
            pipeline_state.log_phase_transition,
            catalog,
            onboard_run_id,
            PHASE_SMA_GATE_1,
            "running",
        )
    if (
        resume_from == "start"
        and (institution_id or "").strip()
        and (input_file_paths_json or "").strip()
    ):
        _state_safe(
            "pipeline_runs input_file_paths (SMA onboard)",
            pipeline_state.update_onboard_pipeline_run_input_file_paths,
            catalog,
            str(institution_id).strip(),
            onboard_run_id,
            str(input_file_paths_json).strip(),
        )


def after_sma_onboard_start(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    cohort_path: Path,
    course_path: Path,
) -> None:
    c, co = cohort_path.as_posix(), course_path.as_posix()
    _state_safe(
        "sma_start -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_START,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (SMA manifest HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (SMA gate)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_1,
        [
            {"artifact_type": "cohort_manifest", "artifact_path": c},
            {"artifact_type": "course_manifest", "artifact_path": co},
        ],
    )


def after_sma_onboard_gate_2_success(catalog: str, institution_id: str, onboard_run_id: str) -> None:
    _state_safe(
        "sma_gate_1 complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_1,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> complete",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "complete",
    )
