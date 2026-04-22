"""Unit tests for :mod:`edvise.genai.mapping.state` (SQL built with ``spark.sql``; mocked Spark)."""

from __future__ import annotations

import pytest

import edvise.genai.mapping.state._sql as sql
import edvise.genai.mapping.state.pipeline_state as pipeline_state
import edvise.genai.mapping.state.table_setup as table_setup


class _Row:
    def __init__(self, d: dict) -> None:
        self._d = d

    def asDict(self) -> dict:  # noqa: N802 (Spark API)
        return self._d


class _Result:
    def __init__(self, values: list[_Row] | None = None) -> None:
        self._v = values or []

    def collect(self) -> list[_Row]:
        return self._v


class _FakeSpark:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self._next_sql_result: _Result = _Result()

    def set_sql_result(self, res: _Result) -> None:
        self._next_sql_result = res

    def sql(self, q: str) -> _Result:  # noqa: ARG002
        self.statements.append(q)
        return self._next_sql_result


def test_lit_escapes_quotes() -> None:
    assert sql.lit("a'b") == "'a''b'"


def test_qualified_table() -> None:
    assert sql.qualified_table("dev", "pipeline_runs") == "`dev`.`genai_mapping`.`pipeline_runs`"


def test_qualified_schema() -> None:
    assert sql.qualified_schema("x") == "`x`.`genai_mapping`"


def test_create_pipeline_run_sql(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.create_pipeline_run("c1", "inst1", "run-1")
    assert len(fake.statements) == 1
    q = fake.statements[0]
    assert "INSERT INTO `c1`.`genai_mapping`.`pipeline_runs`" in q
    assert "`c1`" in q
    assert "'run-1'" in q
    assert "running" in q


def test_get_latest_converts_row(monkeypatch) -> None:
    fake = _FakeSpark()
    d = {
        "institution_id": "i",
        "pipeline_run_id": "r",
        "catalog": "c",
        "status": "complete",
        "created_at": None,
        "updated_at": None,
    }
    fake.set_sql_result(_Result([_Row(d)]))
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    out = pipeline_state.get_latest_pipeline_run("c", "i")
    assert out is not None
    assert out["pipeline_run_id"] == "r"
    # Spark would supply timestamps; with None, passes through
    assert out.get("status") == "complete"


def test_check_hitl_resolution_no_rows_false(monkeypatch) -> None:
    fake = _FakeSpark()
    fake.set_sql_result(_Result([_Row({"n_total": 0, "n_approved": 0})]))
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    assert pipeline_state.check_hitl_resolution("c", "p1", "ia") is False


def test_check_hitl_resolution_all_approved_true(monkeypatch) -> None:
    fake = _FakeSpark()
    fake.set_sql_result(_Result([_Row({"n_total": 2, "n_approved": 2})]))
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    assert pipeline_state.check_hitl_resolution("c", "p1", "ia") is True


def test_resolve_hitl_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="status must be"):
        pipeline_state.resolve_hitl("c", "r1", "ia", "grain", "alice", "pending")


def test_table_setup_runs_ddl(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(table_setup, "get_spark_session", lambda: fake)
    table_setup.create_state_tables("my_cat")
    assert any("CREATE SCHEMA" in s for s in fake.statements)
    assert any("CREATE TABLE" in s and "pipeline_runs" in s for s in fake.statements)
    assert any("hitl_reviews" in s for s in fake.statements)
