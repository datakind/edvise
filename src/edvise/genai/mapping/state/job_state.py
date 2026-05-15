"""
Best-effort pipeline state updates for Databricks job entrypoints (IA / SMA).

Failures are logged and do not block the job (same spirit as
:func:`~edvise.genai.mapping.shared.pipeline_artifacts.merge_genai_pipeline_artifact_rows`).

UC HITL polling helpers (:func:`wait_for_ia_gate_1_hitl`, :func:`wait_for_ia_gate_1_hooks_hitl`,
:func:`wait_for_sma_gate_1_hitl`, :func:`wait_for_sma_gate_2_transformation_review_hitl`,
:func:`wait_for_sma_gate_2_hook_preview_hitl`,
:func:`wait_for_sma_gate_2_hook_required_hitl`,
:func:`wait_for_sma_gate_2_grain_hitl`) are blocking and
raise on timeout or rejection. Timeouts persist
``timed_out`` on ``pipeline_runs`` / ``pipeline_phases`` (resumable); other failures may use
:func:`mark_pipeline_failed`.
"""

from __future__ import annotations

import json
import logging
import typing as t
from pathlib import Path

from edvise.genai.mapping.identity_agent.hitl.schemas import InstitutionHITLItems
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
    InstitutionSMAHITLItems,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas import (
    InstitutionSMATransformationHookHITLItems,
    TransformationReviewHITLFile,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.state import pipeline_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    poll_uc_hitl_until_approved_or_timeout,
)

LOGGER = logging.getLogger(__name__)

PHASE_IA_START: str = "ia_start"
PHASE_IA_GATE_1: str = "ia_gate_1"
PHASE_IA_GATE_1_HOOKS: str = "ia_gate_1_hooks"
PHASE_SMA_START: str = "sma_start"
PHASE_SMA_GATE_1: str = "sma_gate_1"
PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW: str = "sma_gate_2_transformation_review"
PHASE_SMA_GATE_2_HOOK_PREVIEW: str = "sma_gate_2_hook_preview"
PHASE_SMA_GATE_2_HOOK_REQUIRED: str = "sma_gate_2_hook_required"
PHASE_SMA_GATE_2_GRAIN: str = "sma_gate_2_grain"
AUTO_APPROVER: str = "pipeline_auto_approve_empty_hitl"


def _state_safe(
    label: str, fn: t.Callable[..., object], *args: object, **kwargs: object
) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — intentional non-fatal
        LOGGER.warning("Pipeline state [%s] skipped: %s", label, e)


def _hitl_artifact_has_actionable_items(
    artifact_type: str, artifact_path: Path
) -> bool:
    """
    Return True when the artifact contains at least one gate-blocking item.

    IA grain/term: any item with ``choice`` unset.
    SMA cohort/course manifests: any item in ``gate_pending``.
    """
    at = str(artifact_type).strip().lower()
    if at in {"grain", "term"}:
        env_ia = read_pydantic_json(Path(artifact_path), InstitutionHITLItems)
        return len(env_ia.pending) > 0
    if at in {"cohort_manifest", "course_manifest"}:
        env_sma = read_pydantic_json(Path(artifact_path), InstitutionSMAHITLItems)
        return len(env_sma.gate_pending) > 0
    if at in {
        "cohort_transformation_hook_hitl",
        "course_transformation_hook_hitl",
    }:
        env_hooks = read_pydantic_json(
            Path(artifact_path), InstitutionSMATransformationHookHITLItems
        )
        return len(env_hooks.pending) > 0
    if at in {"cohort_transformation_review", "course_transformation_review"}:
        env_review = read_pydantic_json(
            Path(artifact_path), TransformationReviewHITLFile
        )
        return len(env_review.pending) > 0
    if at in {"cohort_sma_grain_hitl", "course_sma_grain_hitl"}:
        env_grain = read_pydantic_json(Path(artifact_path), InstitutionHITLItems)
        return len(env_grain.pending) > 0
    return True


def _auto_approve_hitl_artifact_if_empty(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    artifact_path: Path,
) -> None:
    """
    Auto-approve a UC ``hitl_reviews`` artifact row when the file has no actionable items.
    """
    try:
        has_actionable = _hitl_artifact_has_actionable_items(
            artifact_type, artifact_path
        )
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "Could not inspect HITL artifact for auto-approve: run=%s phase=%s artifact_type=%s path=%s (%s)",
            onboard_run_id,
            phase,
            artifact_type,
            artifact_path,
            e,
        )
        return
    if has_actionable:
        return
    _state_safe(
        f"auto-approve empty HITL artifact ({artifact_type})",
        pipeline_state.resolve_hitl,
        catalog,
        onboard_run_id,
        phase,
        artifact_type,
        AUTO_APPROVER,
        "approved",
    )


def mark_pipeline_failed(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
    _state_safe(
        "update_pipeline_run_status(failed)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "failed",
    )


def after_ia_onboard_start(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    grain_path: Path,
    term_path: Path,
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
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1,
        "grain",
        grain_path,
    )
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1,
        "term",
        term_path,
    )


def _hook_preview_specs_nonempty(artifact_path: Path) -> bool:
    """Return True when the preview JSON has a non-empty ``specs`` list (needs human review)."""
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    specs = data.get("specs")
    return isinstance(specs, list) and len(specs) > 0


def _auto_approve_hook_preview_if_empty(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    artifact_path: Path,
) -> None:
    """Approve UC when the hook preview file has no generated specs."""
    try:
        needs_review = _hook_preview_specs_nonempty(artifact_path)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "Could not inspect hook preview for auto-approve: run=%s phase=%s artifact_type=%s path=%s (%s)",
            onboard_run_id,
            phase,
            artifact_type,
            artifact_path,
            e,
        )
        return
    if needs_review:
        return
    _state_safe(
        f"auto-approve empty hook preview ({artifact_type})",
        pipeline_state.resolve_hitl,
        catalog,
        onboard_run_id,
        phase,
        artifact_type,
        AUTO_APPROVER,
        "approved",
    )


def register_ia_gate_1_hook_preview_artifacts(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    grain_hook_preview_path: Path,
    term_hook_preview_path: Path,
) -> None:
    """
    Register grain + term hook preview JSON paths under ``ia_gate_1_hooks`` and optional auto-approve.

    Preview files are produced after hook-generation LLM calls and before ``apply_hook_spec`` /
    materialize. Rows with empty ``specs`` are auto-approved like empty grain/term HITL artifacts.
    """
    g = grain_hook_preview_path.as_posix()
    t = term_hook_preview_path.as_posix()
    _state_safe(
        "ia_gate_1_hooks -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1_HOOKS,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (IA hook preview)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (ia_gate_1_hooks targets)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1_HOOKS,
        [
            {"artifact_type": "grain_hook_preview", "artifact_path": g},
            {"artifact_type": "term_hook_preview", "artifact_path": t},
        ],
    )
    _auto_approve_hook_preview_if_empty(
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1_HOOKS,
        "grain_hook_preview",
        grain_hook_preview_path,
    )
    _auto_approve_hook_preview_if_empty(
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1_HOOKS,
        "term_hook_preview",
        term_hook_preview_path,
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


def wait_for_ia_gate_1_hooks_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """
    Block until every ``hitl_reviews`` row for ``ia_gate_1_hooks`` is ``approved``.

    Used in IA onboard ``gate_1`` after hook-generation LLM output is written to preview JSON;
    reviewers approve before ``apply_hook_spec`` / materialize / enriched contract build.
    """
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_IA_GATE_1_HOOKS,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def after_ia_onboard_gate_1_hooks_approved(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
    """Log hook-preview gate complete and set pipeline run status back to ``running``."""
    _state_safe(
        "ia_gate_1_hooks complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_IA_GATE_1_HOOKS,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> running (post hook-preview HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "running",
    )


def register_sma_gate_2_transformation_review_artifacts(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    cohort_transformation_review_path: Path,
    course_transformation_review_path: Path,
) -> None:
    """
    Register Step 2b ``review_required`` review JSON under ``sma_gate_2_transformation_review``.

    Artifact types: ``cohort_transformation_review``, ``course_transformation_review``.
    Empty ``items`` lists auto-approve like other SMA HITL artifacts.
    """
    c = cohort_transformation_review_path.as_posix()
    co = course_transformation_review_path.as_posix()
    _state_safe(
        "sma_gate_2_transformation_review -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (SMA transformation review HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (SMA gate 2 transformation review)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        [
            {"artifact_type": "cohort_transformation_review", "artifact_path": c},
            {"artifact_type": "course_transformation_review", "artifact_path": co},
        ],
    )
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        "cohort_transformation_review",
        cohort_transformation_review_path,
    )
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        "course_transformation_review",
        course_transformation_review_path,
    )


def wait_for_sma_gate_2_transformation_review_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """Block until UC rows for ``sma_gate_2_transformation_review`` are approved."""
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def after_sma_gate_2_transformation_review_approved(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
    """Log transformation-review gate complete and set pipeline run status to ``running``."""
    _state_safe(
        "sma_gate_2_transformation_review complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> running (post SMA transformation review HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "running",
    )


def register_sma_gate_2_hook_preview_artifacts(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    cohort_transformation_hook_preview_path: Path,
    course_transformation_hook_preview_path: Path,
) -> None:
    """
    Register SMA Step 2b transform HookSpec preview JSON under ``sma_gate_2_hook_preview``.

    Empty ``specs`` lists auto-approve like IA ``grain_hook_preview`` / ``term_hook_preview``.
    """
    c = cohort_transformation_hook_preview_path.as_posix()
    co = course_transformation_hook_preview_path.as_posix()
    _state_safe(
        "sma_gate_2_hook_preview -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_PREVIEW,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (SMA transform hook preview)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (SMA gate 2 hook preview)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_PREVIEW,
        [
            {"artifact_type": "cohort_transformation_hook_preview", "artifact_path": c},
            {
                "artifact_type": "course_transformation_hook_preview",
                "artifact_path": co,
            },
        ],
    )
    _auto_approve_hook_preview_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_PREVIEW,
        "cohort_transformation_hook_preview",
        cohort_transformation_hook_preview_path,
    )
    _auto_approve_hook_preview_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_PREVIEW,
        "course_transformation_hook_preview",
        course_transformation_hook_preview_path,
    )


def wait_for_sma_gate_2_hook_preview_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """Block until UC rows for ``sma_gate_2_hook_preview`` are approved."""
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_PREVIEW,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def after_sma_gate_2_hook_preview_approved(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
    """Log SMA transform hook-preview gate complete and set pipeline run status to ``running``."""
    _state_safe(
        "sma_gate_2_hook_preview complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_PREVIEW,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> running (post SMA transform hook preview)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "running",
    )


def register_sma_gate_2_hook_required_artifacts(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    cohort_transformation_hook_hitl_path: Path,
    course_transformation_hook_hitl_path: Path,
) -> None:
    """
    Register Step 2b ``hook_required`` review JSON paths under ``sma_gate_2_hook_required``.

    Empty ``items`` lists auto-approve like empty SMA manifest HITL artifacts.
    """
    c = cohort_transformation_hook_hitl_path.as_posix()
    co = course_transformation_hook_hitl_path.as_posix()
    _state_safe(
        "sma_gate_2_hook_required -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_REQUIRED,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (SMA transformation hook HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (SMA gate 2 hook_required)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_REQUIRED,
        [
            {"artifact_type": "cohort_transformation_hook_hitl", "artifact_path": c},
            {"artifact_type": "course_transformation_hook_hitl", "artifact_path": co},
        ],
    )
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_REQUIRED,
        "cohort_transformation_hook_hitl",
        cohort_transformation_hook_hitl_path,
    )
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_REQUIRED,
        "course_transformation_hook_hitl",
        course_transformation_hook_hitl_path,
    )


def wait_for_sma_gate_2_hook_required_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """Block until UC rows for ``sma_gate_2_hook_required`` are approved."""
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_REQUIRED,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def after_sma_gate_2_hook_required_approved(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
    """Log transformation-hook gate complete and set pipeline run status back to ``running``."""
    _state_safe(
        "sma_gate_2_hook_required complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_HOOK_REQUIRED,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> running (post SMA transformation hook HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "running",
    )


def _sma_grain_hitl_artifact_row(path: Path) -> dict[str, str]:
    name = path.name.lower()
    if name == "cohort_sma_grain_hitl.json":
        return {
            "artifact_type": "cohort_sma_grain_hitl",
            "artifact_path": path.as_posix(),
        }
    if name == "course_sma_grain_hitl.json":
        return {
            "artifact_type": "course_sma_grain_hitl",
            "artifact_path": path.as_posix(),
        }
    raise ValueError(
        f"Unrecognized SMA grain HITL filename (expected cohort|course): {path}"
    )


def register_sma_gate_2_grain_artifacts(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    grain_hitl_paths: list[Path],
) -> None:
    """
    Register ``InstitutionHITLItems`` SMA grain JSON under ``sma_gate_2_grain``.

    Artifact types: ``cohort_sma_grain_hitl``, ``course_sma_grain_hitl``.
    """
    paths = [Path(p) for p in grain_hitl_paths]
    if not paths:
        return
    artifacts = [_sma_grain_hitl_artifact_row(p) for p in paths]
    _state_safe(
        "sma_gate_2_grain -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_GRAIN,
        "awaiting_hitl",
    )
    _state_safe(
        "pipeline_runs -> awaiting_hitl (SMA grain reconciliation HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        "register_hitl (SMA gate 2 grain)",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_GRAIN,
        artifacts,
    )
    for p in paths:
        row = _sma_grain_hitl_artifact_row(p)
        _auto_approve_hitl_artifact_if_empty(
            catalog,
            onboard_run_id,
            PHASE_SMA_GATE_2_GRAIN,
            row["artifact_type"],
            p,
        )


def wait_for_sma_gate_2_grain_hitl(
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """Block until UC rows for ``sma_gate_2_grain`` are approved."""
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        PHASE_SMA_GATE_2_GRAIN,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def after_sma_gate_2_grain_approved(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
    """Log SMA grain gate complete and set pipeline run status back to ``running``."""
    _state_safe(
        "sma_gate_2_grain complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_2_GRAIN,
        "complete",
    )
    _state_safe(
        "pipeline_runs -> running (post SMA grain HITL)",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "running",
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


def after_ia_onboard_gate_1_success(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
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
        "upsert_onboard_pipeline_run_row",
        pipeline_state.upsert_onboard_pipeline_run_row,
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
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_1,
        "cohort_manifest",
        cohort_path,
    )
    _auto_approve_hitl_artifact_if_empty(
        catalog,
        onboard_run_id,
        PHASE_SMA_GATE_1,
        "course_manifest",
        course_path,
    )


def after_sma_onboard_gate_2_success(
    catalog: str, institution_id: str, onboard_run_id: str
) -> None:
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
