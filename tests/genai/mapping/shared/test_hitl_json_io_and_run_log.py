"""Tests for shared HITL JSON I/O, UTC helper, and run_log append."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from edvise.genai.mapping.shared.hitl.json_io import (
    read_pydantic_json,
    write_pydantic_json,
)
from edvise.genai.mapping.shared.hitl.run_log import (
    RunEvent,
    RunLog,
    SMARRunEvent,
    append_run_log_event,
    resolve_task_run_id,
)
from edvise.genai.mapping.shared.hitl.time import utc_now_iso


class _Sample(BaseModel):
    x: int


def test_write_read_pydantic_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    write_pydantic_json(p, _Sample(x=1))
    assert read_pydantic_json(p, _Sample).x == 1


def test_read_pydantic_json_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_pydantic_json(tmp_path / "nope.json", _Sample)


def test_utc_now_iso_format() -> None:
    s = utc_now_iso()
    assert "T" in s
    assert s.endswith("+00:00")


def test_append_run_log_mixed_events(tmp_path: Path) -> None:
    log_path = tmp_path / "run_log.json"
    append_run_log_event(
        log_path,
        "inst_a",
        RunEvent(
            timestamp="t1",
            resolved_by=None,
            agent="identity_agent",
            domain="grain",
            item_id="i1",
            choice=1,
            option_id="opt",
            reentry="terminal",
        ),
    )
    append_run_log_event(
        log_path,
        "inst_a",
        SMARRunEvent(
            timestamp="t2",
            resolved_by="r",
            agent="schema_mapping_agent",
            entity_type="course",
            item_id="i2",
            target_field="term",
            failure_mode="low_confidence",
            choice=2,
            option_id="b",
            reentry="terminal",
        ),
    )
    loaded = read_pydantic_json(log_path, RunLog)
    assert loaded.institution_id == "inst_a"
    assert len(loaded.events) == 2
    assert isinstance(loaded.events[0], RunEvent)
    assert isinstance(loaded.events[1], SMARRunEvent)
    assert loaded.events[1].target_field == "term"


def test_resolve_task_run_id_prefers_task_specific_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABRICKS_TASK_RUN_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_RUN_ID", raising=False)
    monkeypatch.setenv("DATABRICKS_TASK_RUN_ID", "task-env")
    monkeypatch.setenv("DATABRICKS_RUN_ID", "run-env")
    assert resolve_task_run_id() == "task-env"


def test_append_run_log_event_injects_task_run_id_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABRICKS_TASK_RUN_ID", "tr-999")
    log_path = tmp_path / "run_log.json"
    append_run_log_event(
        log_path,
        "inst_a",
        RunEvent(
            timestamp="t1",
            resolved_by=None,
            agent="identity_agent",
            domain="grain",
            item_id="i1",
            choice=1,
            option_id="opt",
            reentry="terminal",
        ),
    )
    loaded = read_pydantic_json(log_path, RunLog)
    assert loaded.events[0].task_run_id == "tr-999"


def test_append_run_log_event_keeps_explicit_task_run_id_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABRICKS_TASK_RUN_ID", "from-env")
    log_path = tmp_path / "run_log.json"
    append_run_log_event(
        log_path,
        "inst_a",
        RunEvent(
            timestamp="t1",
            resolved_by=None,
            agent="identity_agent",
            domain="grain",
            item_id="i1",
            choice=1,
            option_id="opt",
            reentry="terminal",
            task_run_id="explicit-id",
        ),
    )
    loaded = read_pydantic_json(log_path, RunLog)
    assert loaded.events[0].task_run_id == "explicit-id"
