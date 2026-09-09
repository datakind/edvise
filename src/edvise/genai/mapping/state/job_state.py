"""
Best-effort pipeline state updates for Databricks job entrypoints (IA / SMA).

Failures are logged and do not block the job (same spirit as
:func:`~edvise.genai.mapping.shared.pipeline_artifacts.merge_genai_pipeline_artifact_rows`).

Each UC HITL gate is declared once as a :class:`HitlGate` in the gate table below: its phase,
how its artifacts auto-approve when empty, and — for the two ``*_start`` registrations — which
phase goes to ``awaiting_hitl`` while artifacts register under the *following* gate.

A gate is then passed as a value through the three things you do to it::

    job_state.register_ia_gate_1_hook_preview_artifacts(...)          # queue for review
    job_state.wait_for_gate(job_state.GATE_IA_1_HOOKS, catalog, run_id, institution_id=...)
    job_state.complete_gate(job_state.GATE_IA_1_HOOKS, catalog, institution_id, run_id)

:func:`wait_for_gate` is blocking and raises on timeout or rejection. Timeouts persist
``timed_out`` on ``pipeline_runs`` / ``pipeline_phases`` (resumable); other failures may use
:func:`mark_pipeline_failed`.

Registration keeps a named wrapper per gate because each one owns the mapping from its file
paths to UC ``artifact_type`` names — vocabulary that belongs here rather than in pipeline code.
"""

from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass
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

# (catalog, onboard_run_id, phase, artifact_type, artifact_path) -> None
AutoApproveFn = t.Callable[[str, str, str, str, Path], None]


def _state_safe(
    label: str, fn: t.Callable[..., object], *args: object, **kwargs: object
) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — intentional non-fatal
        LOGGER.warning("Pipeline state [%s] skipped: %s", label, e)


# ---------------------------------------------------------------------------
# Auto-approve strategies — how a gate decides an artifact needs no reviewer
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gate table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HitlGate:
    """
    One UC HITL gate.

    ``phase`` is where artifacts register and what reviewers approve against.
    ``awaiting_phase`` is the phase marked ``awaiting_hitl`` at registration; it differs from
    ``phase`` only for the two ``*_start`` registrations, which end the ``*_start`` phase while
    queueing work for the *following* gate.
    """

    phase: str
    auto_approve: AutoApproveFn = _auto_approve_hitl_artifact_if_empty
    awaiting_phase: str | None = None

    @property
    def awaiting(self) -> str:
        return self.awaiting_phase or self.phase


# Grain/term HITL JSON written by IA onboard start; awaited at the top of IA onboard
# ``resume_from=gate_1`` before the local JSON gates are checked.
GATE_IA_1 = HitlGate(PHASE_IA_GATE_1, awaiting_phase=PHASE_IA_START)

# HookSpec previews from IA's hook-generation LLM; reviewers approve before
# ``apply_hook_spec`` / materialize / enriched contract build.
GATE_IA_1_HOOKS = HitlGate(
    PHASE_IA_GATE_1_HOOKS, auto_approve=_auto_approve_hook_preview_if_empty
)

# Cohort/course mapping manifests from SMA Step 2a; awaited at the top of SMA onboard
# ``resume_from=gate_2`` before manifest HITL JSON is resolved on disk.
GATE_SMA_1 = HitlGate(PHASE_SMA_GATE_1, awaiting_phase=PHASE_SMA_START)

# Step 2b plans flagged ``review_required``.
GATE_SMA_2_TRANSFORMATION_REVIEW = HitlGate(PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW)

# Step 2b transform HookSpec previews, before ``transform_hooks.py`` is materialized.
GATE_SMA_2_HOOK_PREVIEW = HitlGate(
    PHASE_SMA_GATE_2_HOOK_PREVIEW, auto_approve=_auto_approve_hook_preview_if_empty
)

# Deprecated: hook disposition now happens in the transformation review gate. Kept so
# in-flight UC rows still resolve.
GATE_SMA_2_HOOK_REQUIRED = HitlGate(PHASE_SMA_GATE_2_HOOK_REQUIRED)

# Raised when manifest grain is stricter than the cleaned row count.
GATE_SMA_2_GRAIN = HitlGate(PHASE_SMA_GATE_2_GRAIN)


# ---------------------------------------------------------------------------
# Operations on a gate
# ---------------------------------------------------------------------------


def _register_gate(
    gate: HitlGate,
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    artifacts: dict[str, Path],
) -> None:
    """
    Queue ``gate`` for review: mark the outgoing phase ``awaiting_hitl``, flip ``pipeline_runs``
    to ``awaiting_hitl``, register ``artifacts`` (artifact type -> path), then auto-approve the
    ones with nothing actionable in them.
    """
    _state_safe(
        f"{gate.awaiting} -> awaiting_hitl",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        gate.awaiting,
        "awaiting_hitl",
    )
    _state_safe(
        f"pipeline_runs -> awaiting_hitl ({gate.phase})",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        "awaiting_hitl",
    )
    _state_safe(
        f"register_hitl ({gate.phase})",
        pipeline_state.register_hitl_artifacts,
        catalog,
        onboard_run_id,
        gate.phase,
        [
            {"artifact_type": artifact_type, "artifact_path": path.as_posix()}
            for artifact_type, path in artifacts.items()
        ],
    )
    for artifact_type, path in artifacts.items():
        gate.auto_approve(catalog, onboard_run_id, gate.phase, artifact_type, path)


def wait_for_gate(
    gate: HitlGate,
    catalog: str,
    onboard_run_id: str,
    *,
    institution_id: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """
    Block until every ``hitl_reviews`` row for ``gate`` is ``approved`` in Unity Catalog.

    Blocking; raises on timeout or rejection.
    """
    return poll_uc_hitl_until_approved_or_timeout(
        catalog,
        institution_id,
        onboard_run_id,
        gate.phase,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )


def complete_gate(
    gate: HitlGate,
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    run_status: str = "running",
) -> None:
    """
    Mark ``gate`` complete and move ``pipeline_runs`` on to ``run_status``.

    ``run_status`` stays ``running`` for every gate except the last one in an onboard run,
    which passes ``"complete"`` to end the run.
    """
    _state_safe(
        f"{gate.phase} complete",
        pipeline_state.log_phase_transition,
        catalog,
        onboard_run_id,
        gate.phase,
        "complete",
    )
    _state_safe(
        f"pipeline_runs -> {run_status} (post {gate.phase})",
        pipeline_state.update_pipeline_run_status,
        catalog,
        institution_id,
        onboard_run_id,
        run_status,
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


# --- IA --------------------------------------------------------------------


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


def after_ia_onboard_start(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    grain_path: Path,
    term_path: Path,
) -> None:
    _register_gate(
        GATE_IA_1,
        catalog,
        institution_id,
        onboard_run_id,
        {"grain": grain_path, "term": term_path},
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
    _register_gate(
        GATE_IA_1_HOOKS,
        catalog,
        institution_id,
        onboard_run_id,
        {
            "grain_hook_preview": grain_hook_preview_path,
            "term_hook_preview": term_hook_preview_path,
        },
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
    reference_id: str | None = None,
    reference_content_hash: str | None = None,
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
    if (
        resume_from == "start"
        and (institution_id or "").strip()
        and (reference_id or "").strip()
    ):
        _state_safe(
            "pipeline_runs reference_id (SMA onboard)",
            pipeline_state.update_onboard_pipeline_run_reference,
            catalog,
            str(institution_id).strip(),
            onboard_run_id,
            reference_id=str(reference_id).strip(),
            reference_content_hash=(
                str(reference_content_hash).strip()
                if (reference_content_hash or "").strip()
                else None
            ),
        )


def after_sma_onboard_start(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    *,
    cohort_path: Path,
    course_path: Path,
) -> None:
    _register_gate(
        GATE_SMA_1,
        catalog,
        institution_id,
        onboard_run_id,
        {"cohort_manifest": cohort_path, "course_manifest": course_path},
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
    _register_gate(
        GATE_SMA_2_TRANSFORMATION_REVIEW,
        catalog,
        institution_id,
        onboard_run_id,
        {
            "cohort_transformation_review": cohort_transformation_review_path,
            "course_transformation_review": course_transformation_review_path,
        },
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
    _register_gate(
        GATE_SMA_2_HOOK_PREVIEW,
        catalog,
        institution_id,
        onboard_run_id,
        {
            "cohort_transformation_hook_preview": cohort_transformation_hook_preview_path,
            "course_transformation_hook_preview": course_transformation_hook_preview_path,
        },
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

    **Deprecated:** ``edvise_genai_sma`` no longer calls this; hook disposition is handled in
    ``sma_gate_2_transformation_review`` (option ``hook_required``). Kept for in-flight UC rows.

    Empty ``items`` lists auto-approve like empty SMA manifest HITL artifacts.
    """
    _register_gate(
        GATE_SMA_2_HOOK_REQUIRED,
        catalog,
        institution_id,
        onboard_run_id,
        {
            "cohort_transformation_hook_hitl": cohort_transformation_hook_hitl_path,
            "course_transformation_hook_hitl": course_transformation_hook_hitl_path,
        },
    )


def _sma_grain_artifact_type(path: Path) -> str:
    """Map a SMA grain HITL filename to its UC ``artifact_type``."""
    name = path.name.lower()
    if name == "cohort_sma_grain_hitl.json":
        return "cohort_sma_grain_hitl"
    if name == "course_sma_grain_hitl.json":
        return "course_sma_grain_hitl"
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
    _register_gate(
        GATE_SMA_2_GRAIN,
        catalog,
        institution_id,
        onboard_run_id,
        {_sma_grain_artifact_type(p): p for p in paths},
    )
