"""
Unit tests for :mod:`edvise.genai.mapping.state.job_state`.

The gate table (:class:`HitlGate` constants) is the declarative source of truth for each UC
HITL gate, and :func:`_register_gate` / :func:`complete_gate` are the two shared multi-step
operations on a gate. These tests cover the table itself, both shared operations, and the
public per-gate wrappers that pipeline entry points call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

import edvise.genai.mapping.state.job_state as job_state
from edvise.genai.mapping.state import pipeline_state


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def make(self, name: str) -> Callable[..., None]:
        def _fn(*args: Any, **kwargs: Any) -> None:
            self.calls.append((name, args, kwargs))

        return _fn

    def call_names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def args_for(self, name: str) -> tuple[Any, ...]:
        for called, args, _ in self.calls:
            if called == name:
                return args
        raise AssertionError(f"{name} was never called; got {self.call_names()}")


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(
        pipeline_state, "log_phase_transition", rec.make("log_phase_transition")
    )
    monkeypatch.setattr(
        pipeline_state,
        "update_pipeline_run_status",
        rec.make("update_pipeline_run_status"),
    )
    monkeypatch.setattr(
        pipeline_state, "register_hitl_artifacts", rec.make("register_hitl_artifacts")
    )
    monkeypatch.setattr(pipeline_state, "resolve_hitl", rec.make("resolve_hitl"))
    return rec


# ---------------------------------------------------------------------------
# Gate table — the declarative spec every wrapper reads from
# ---------------------------------------------------------------------------

_GATE_TABLE_CASES = [
    # gate, phase, awaiting phase, auto-approve strategy
    (
        job_state.GATE_IA_1,
        job_state.PHASE_IA_GATE_1,
        job_state.PHASE_IA_START,
        job_state._auto_approve_hitl_artifact_if_empty,
    ),
    (
        job_state.GATE_IA_1_HOOKS,
        job_state.PHASE_IA_GATE_1_HOOKS,
        job_state.PHASE_IA_GATE_1_HOOKS,
        job_state._auto_approve_hook_preview_if_empty,
    ),
    (
        job_state.GATE_SMA_1,
        job_state.PHASE_SMA_GATE_1,
        job_state.PHASE_SMA_START,
        job_state._auto_approve_hitl_artifact_if_empty,
    ),
    (
        job_state.GATE_SMA_2_TRANSFORMATION_REVIEW,
        job_state.PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        job_state.PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
        job_state._auto_approve_hitl_artifact_if_empty,
    ),
    (
        job_state.GATE_SMA_2_HOOK_PREVIEW,
        job_state.PHASE_SMA_GATE_2_HOOK_PREVIEW,
        job_state.PHASE_SMA_GATE_2_HOOK_PREVIEW,
        job_state._auto_approve_hook_preview_if_empty,
    ),
    (
        job_state.GATE_SMA_2_HOOK_REQUIRED,
        job_state.PHASE_SMA_GATE_2_HOOK_REQUIRED,
        job_state.PHASE_SMA_GATE_2_HOOK_REQUIRED,
        job_state._auto_approve_hitl_artifact_if_empty,
    ),
    (
        job_state.GATE_SMA_2_GRAIN,
        job_state.PHASE_SMA_GATE_2_GRAIN,
        job_state.PHASE_SMA_GATE_2_GRAIN,
        job_state._auto_approve_hitl_artifact_if_empty,
    ),
]


@pytest.mark.parametrize("gate,phase,awaiting,auto_approve", _GATE_TABLE_CASES)
def test_gate_table(
    gate: job_state.HitlGate,
    phase: str,
    awaiting: str,
    auto_approve: job_state.AutoApproveFn,
) -> None:
    """Only the two ``*_start`` gates await a different phase than they register under."""
    assert gate.phase == phase
    assert gate.awaiting == awaiting
    # Hook-preview gates check a ``specs`` list; everything else checks HITL items.
    assert gate.auto_approve is auto_approve


# ---------------------------------------------------------------------------
# _register_gate — shared registration body
# ---------------------------------------------------------------------------


def test_register_gate_registers_rows_then_auto_approves_each(
    recorder: _Recorder, tmp_path: Path
) -> None:
    seen: list[tuple] = []
    gate = job_state.HitlGate(
        "test_phase",
        auto_approve=lambda *args: seen.append(args),
        awaiting_phase="test_awaiting",
    )
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"

    job_state._register_gate(
        gate, "cat", "inst1", "run1", {"type_a": path_a, "type_b": path_b}
    )

    assert recorder.call_names() == [
        "log_phase_transition",
        "update_pipeline_run_status",
        "register_hitl_artifacts",
    ]
    assert recorder.args_for("log_phase_transition") == (
        "cat",
        "run1",
        "test_awaiting",
        "awaiting_hitl",
    )
    assert recorder.args_for("update_pipeline_run_status") == (
        "cat",
        "inst1",
        "run1",
        "awaiting_hitl",
    )
    register_args = recorder.args_for("register_hitl_artifacts")
    assert register_args[:3] == ("cat", "run1", "test_phase")
    assert register_args[3] == [
        {"artifact_type": "type_a", "artifact_path": path_a.as_posix()},
        {"artifact_type": "type_b", "artifact_path": path_b.as_posix()},
    ]
    # Auto-approve runs against the gate phase, once per artifact, after registration.
    assert seen == [
        ("cat", "run1", "test_phase", "type_a", path_a),
        ("cat", "run1", "test_phase", "type_b", path_b),
    ]


# ---------------------------------------------------------------------------
# wait_for_gate — polls UC for whichever gate it's handed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate,expected_phase,_awaiting,_auto", _GATE_TABLE_CASES)
def test_wait_for_gate_polls_that_gates_phase(
    monkeypatch: pytest.MonkeyPatch,
    gate: job_state.HitlGate,
    expected_phase: str,
    _awaiting: str,
    _auto: job_state.AutoApproveFn,
) -> None:
    calls: list[tuple] = []

    def _fake_poll(
        catalog: str,
        institution_id: str,
        onboard_run_id: str,
        phase: str,
        **kwargs: Any,
    ) -> bool:
        calls.append((catalog, institution_id, onboard_run_id, phase, kwargs))
        return True

    monkeypatch.setattr(job_state, "poll_uc_hitl_until_approved_or_timeout", _fake_poll)

    assert job_state.wait_for_gate(gate, "cat", "run1", institution_id="inst1") is True

    assert len(calls) == 1
    catalog, institution_id, onboard_run_id, phase, kwargs = calls[0]
    assert (catalog, institution_id, onboard_run_id, phase) == (
        "cat",
        "inst1",
        "run1",
        expected_phase,
    )
    assert (
        kwargs["poll_interval_seconds"] == job_state.DEFAULT_HITL_POLL_INTERVAL_SECONDS
    )
    assert kwargs["timeout_seconds"] == job_state.DEFAULT_HITL_POLL_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# complete_gate — defaults to resuming the run; terminal status is opt-in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate,expected_phase,_awaiting,_auto", _GATE_TABLE_CASES)
def test_complete_gate_completes_that_gates_phase(
    recorder: _Recorder,
    gate: job_state.HitlGate,
    expected_phase: str,
    _awaiting: str,
    _auto: job_state.AutoApproveFn,
) -> None:
    job_state.complete_gate(gate, "cat", "inst1", "run1")

    assert recorder.args_for("log_phase_transition") == (
        "cat",
        "run1",
        expected_phase,
        "complete",
    )
    assert recorder.args_for("update_pipeline_run_status") == (
        "cat",
        "inst1",
        "run1",
        "running",
    )


def test_complete_gate_can_end_the_run(recorder: _Recorder) -> None:
    """SMA's final gate ends the onboard run instead of resuming it."""
    job_state.complete_gate(
        job_state.GATE_SMA_1, "cat", "inst1", "run1", run_status="complete"
    )

    assert recorder.args_for("log_phase_transition") == (
        "cat",
        "run1",
        job_state.PHASE_SMA_GATE_1,
        "complete",
    )
    assert recorder.args_for("update_pipeline_run_status") == (
        "cat",
        "inst1",
        "run1",
        "complete",
    )


# ---------------------------------------------------------------------------
# register_* wrappers — correct gate, correct artifact types
# ---------------------------------------------------------------------------


def _registered(recorder: _Recorder) -> tuple[str, str, list[dict[str, str]]]:
    """(awaiting phase, gate phase, artifact rows) from a register_* call."""
    awaiting = recorder.args_for("log_phase_transition")[2]
    register_args = recorder.args_for("register_hitl_artifacts")
    return awaiting, register_args[2], register_args[3]


def test_after_ia_onboard_start_registers_under_gate_1(
    recorder: _Recorder, tmp_path: Path
) -> None:
    grain_path = tmp_path / "identity_grain_hitl.json"
    term_path = tmp_path / "identity_term_hitl.json"

    job_state.after_ia_onboard_start(
        "cat", "inst1", "run1", grain_path=grain_path, term_path=term_path
    )

    awaiting, phase, rows = _registered(recorder)
    # ia_start is the phase that goes awaiting; artifacts belong to the following gate.
    assert (awaiting, phase) == (job_state.PHASE_IA_START, job_state.PHASE_IA_GATE_1)
    assert rows == [
        {"artifact_type": "grain", "artifact_path": grain_path.as_posix()},
        {"artifact_type": "term", "artifact_path": term_path.as_posix()},
    ]


def test_after_sma_onboard_start_registers_under_gate_1(
    recorder: _Recorder, tmp_path: Path
) -> None:
    cohort_path = tmp_path / "cohort_hitl_manifest.json"
    course_path = tmp_path / "course_hitl_manifest.json"

    job_state.after_sma_onboard_start(
        "cat", "inst1", "run1", cohort_path=cohort_path, course_path=course_path
    )

    awaiting, phase, rows = _registered(recorder)
    assert (awaiting, phase) == (job_state.PHASE_SMA_START, job_state.PHASE_SMA_GATE_1)
    assert [r["artifact_type"] for r in rows] == ["cohort_manifest", "course_manifest"]


def test_register_ia_gate_1_hook_preview_artifacts(
    recorder: _Recorder, tmp_path: Path
) -> None:
    grain_path = tmp_path / "identity_grain_hook_preview.json"
    term_path = tmp_path / "identity_term_hook_preview.json"

    job_state.register_ia_gate_1_hook_preview_artifacts(
        "cat",
        "inst1",
        "run1",
        grain_hook_preview_path=grain_path,
        term_hook_preview_path=term_path,
    )

    awaiting, phase, rows = _registered(recorder)
    assert awaiting == phase == job_state.PHASE_IA_GATE_1_HOOKS
    assert rows == [
        {"artifact_type": "grain_hook_preview", "artifact_path": grain_path.as_posix()},
        {"artifact_type": "term_hook_preview", "artifact_path": term_path.as_posix()},
    ]


def test_register_sma_gate_2_transformation_review_artifacts(
    recorder: _Recorder, tmp_path: Path
) -> None:
    cohort_path = tmp_path / "cohort_transformation_review.json"
    course_path = tmp_path / "course_transformation_review.json"

    job_state.register_sma_gate_2_transformation_review_artifacts(
        "cat",
        "inst1",
        "run1",
        cohort_transformation_review_path=cohort_path,
        course_transformation_review_path=course_path,
    )

    awaiting, phase, rows = _registered(recorder)
    assert awaiting == phase == job_state.PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW
    assert rows == [
        {
            "artifact_type": "cohort_transformation_review",
            "artifact_path": cohort_path.as_posix(),
        },
        {
            "artifact_type": "course_transformation_review",
            "artifact_path": course_path.as_posix(),
        },
    ]


def test_register_sma_gate_2_hook_preview_artifacts(
    recorder: _Recorder, tmp_path: Path
) -> None:
    cohort_path = tmp_path / "cohort_transformation_hook_preview.json"
    course_path = tmp_path / "course_transformation_hook_preview.json"

    job_state.register_sma_gate_2_hook_preview_artifacts(
        "cat",
        "inst1",
        "run1",
        cohort_transformation_hook_preview_path=cohort_path,
        course_transformation_hook_preview_path=course_path,
    )

    awaiting, phase, rows = _registered(recorder)
    assert awaiting == phase == job_state.PHASE_SMA_GATE_2_HOOK_PREVIEW
    assert [r["artifact_type"] for r in rows] == [
        "cohort_transformation_hook_preview",
        "course_transformation_hook_preview",
    ]


def test_register_sma_gate_2_hook_required_artifacts(
    recorder: _Recorder, tmp_path: Path
) -> None:
    cohort_path = tmp_path / "cohort_transformation_hook_hitl.json"
    course_path = tmp_path / "course_transformation_hook_hitl.json"

    job_state.register_sma_gate_2_hook_required_artifacts(
        "cat",
        "inst1",
        "run1",
        cohort_transformation_hook_hitl_path=cohort_path,
        course_transformation_hook_hitl_path=course_path,
    )

    awaiting, phase, rows = _registered(recorder)
    assert awaiting == phase == job_state.PHASE_SMA_GATE_2_HOOK_REQUIRED
    assert [r["artifact_type"] for r in rows] == [
        "cohort_transformation_hook_hitl",
        "course_transformation_hook_hitl",
    ]


def test_register_sma_gate_2_grain_artifacts_derives_types_from_filenames(
    recorder: _Recorder, tmp_path: Path
) -> None:
    cohort_path = tmp_path / "cohort_sma_grain_hitl.json"
    course_path = tmp_path / "course_sma_grain_hitl.json"

    job_state.register_sma_gate_2_grain_artifacts(
        "cat", "inst1", "run1", grain_hitl_paths=[cohort_path, course_path]
    )

    awaiting, phase, rows = _registered(recorder)
    assert awaiting == phase == job_state.PHASE_SMA_GATE_2_GRAIN
    assert rows == [
        {
            "artifact_type": "cohort_sma_grain_hitl",
            "artifact_path": cohort_path.as_posix(),
        },
        {
            "artifact_type": "course_sma_grain_hitl",
            "artifact_path": course_path.as_posix(),
        },
    ]


def test_register_sma_gate_2_grain_artifacts_empty_list_is_noop(
    recorder: _Recorder,
) -> None:
    job_state.register_sma_gate_2_grain_artifacts(
        "cat", "inst1", "run1", grain_hitl_paths=[]
    )
    assert recorder.calls == []


def test_sma_grain_artifact_type_rejects_unrecognized_filename() -> None:
    with pytest.raises(ValueError, match="Unrecognized SMA grain HITL filename"):
        job_state._sma_grain_artifact_type(Path("/tmp/unexpected.json"))
