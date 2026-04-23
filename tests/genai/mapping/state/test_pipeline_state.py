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

    def __getitem__(self, key: str):  # Spark Row supports column access by name
        return self._d[key]


class _Result:
    def __init__(self, values: list[_Row] | None = None) -> None:
        self._v = values or []

    def collect(self) -> list[_Row]:
        return self._v


class _FakeSpark:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self._next_sql_result: _Result = _Result()
        self._sql_result_queue: list[_Result] | None = None

    def set_sql_result(self, res: _Result) -> None:
        self._next_sql_result = res
        self._sql_result_queue = None

    def set_sql_result_queue(self, queue: list[_Result]) -> None:
        """Consume one :class:`_Result` per ``sql()`` call in FIFO order."""
        self._sql_result_queue = list(queue)
        self._next_sql_result = _Result()

    def sql(self, q: str) -> _Result:  # noqa: ARG002
        self.statements.append(q)
        if self._sql_result_queue:
            return self._sql_result_queue.pop(0)
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
    assert "onboard_run_id" in q
    assert "execute_run_id" in q
    assert "`c1`" in q
    assert "'run-1'" in q
    assert "running" in q
    assert "db_run_id" in q
    assert "NULL" in q


def test_create_pipeline_run_sql_db_run_id(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.create_pipeline_run("c1", "inst1", "run-1", db_run_id="job-999")
    q = fake.statements[0]
    assert "db_run_id" in q
    assert "'job-999'" in q


def test_create_pipeline_run_sql_input_file_paths(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.create_pipeline_run(
        "c1",
        "inst1",
        "run-1",
        input_file_paths_json='{"cohort":["/Volumes/x/a.csv"]}',
    )
    q = fake.statements[0]
    assert "input_file_paths" in q
    assert "cohort" in q


def test_update_onboard_pipeline_run_input_file_paths(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.update_onboard_pipeline_run_input_file_paths(
        "c1", "inst1", "run-1", '{"a":["/p"]}'
    )
    q = fake.statements[0]
    assert "UPDATE" in q
    assert "execute_run_id IS NULL" in q
    assert "input_file_paths" in q


def test_update_execute_pipeline_run_input_file_paths(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.update_execute_pipeline_run_input_file_paths(
        "c1", "inst1", "ex-1", '{"a":["/p"]}'
    )
    q = fake.statements[0]
    assert "UPDATE" in q
    assert "execute_run_id" in q
    assert "input_file_paths" in q


def test_get_latest_converts_row(monkeypatch) -> None:
    fake = _FakeSpark()
    d = {
        "institution_id": "i",
        "onboard_run_id": "r",
        "catalog": "c",
        "status": "complete",
        "created_at": None,
        "updated_at": None,
    }
    fake.set_sql_result(_Result([_Row(d)]))
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    out = pipeline_state.get_latest_pipeline_run("c", "i")
    assert out is not None
    assert out["onboard_run_id"] == "r"
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
    pr_create = next(s for s in fake.statements if "CREATE TABLE" in s and "pipeline_runs" in s)
    assert "onboard_run_id" in pr_create
    assert "execute_run_id" in pr_create
    assert "db_run_id" in pr_create
    assert any("ALTER TABLE" in s and "db_run_id" in s for s in fake.statements)
    assert any("ALTER TABLE" in s and "execute_run_id" in s for s in fake.statements)
    assert any("ALTER TABLE" in s and "input_file_paths" in s for s in fake.statements)
    assert sum("RENAME COLUMN pipeline_run_id TO onboard_run_id" in s for s in fake.statements) == 3
    assert any("hitl_reviews" in s for s in fake.statements)


def test_resolve_onboard_run_id_explicit_override(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    assert pipeline_state.resolve_onboard_run_id("c", "inst", "  my_run  ") == "my_run"
    assert fake.statements == []


def test_resolve_onboard_run_id_no_row_today(monkeypatch) -> None:
    fake = _FakeSpark()
    fake.set_sql_result_queue([_Result([])])
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    from datetime import date

    base = f"foo_{date.today().strftime('%Y%m%d')}"
    assert pipeline_state.resolve_onboard_run_id("c", "foo", None) == base
    assert "to_date(created_at) = current_date()" in fake.statements[0]


def test_resolve_onboard_run_id_resume_timed_out(monkeypatch) -> None:
    fake = _FakeSpark()
    latest = {
        "institution_id": "foo",
        "onboard_run_id": "foo_20990101",
        "catalog": "c",
        "status": "timed_out",
        "created_at": None,
        "updated_at": None,
    }
    fake.set_sql_result_queue([_Result([_Row(latest)])])
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    assert pipeline_state.resolve_onboard_run_id("c", "foo", None) == "foo_20990101"


def test_resolve_onboard_run_id_new_suffix_after_complete(monkeypatch) -> None:
    fake = _FakeSpark()
    latest = {
        "institution_id": "foo",
        "onboard_run_id": "foo_20990101",
        "catalog": "c",
        "status": "complete",
        "created_at": None,
        "updated_at": None,
    }
    fake.set_sql_result_queue(
        [
            _Result([_Row(latest)]),
            _Result([_Row({"n": 2})]),
        ]
    )
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    from datetime import date

    base = f"foo_{date.today().strftime('%Y%m%d')}"
    assert pipeline_state.resolve_onboard_run_id("c", "foo", None) == f"{base}_3"


def test_reconcile_stale_emits_merge_and_update(monkeypatch) -> None:
    fake = _FakeSpark()
    fake.set_sql_result_queue([_Result([]), _Result([])])
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    monkeypatch.setattr(pipeline_state, "create_state_tables", lambda *args, **kwargs: None)
    pipeline_state.reconcile_stale_nonterminal_pipeline_runs("c1", "inst1", 10)
    assert len(fake.statements) == 2
    assert "MERGE INTO" in fake.statements[0] and "pipeline_phases" in fake.statements[0]
    assert "execute_run_id IS NULL" in fake.statements[0]
    assert "UPDATE" in fake.statements[1] and "timed_out" in fake.statements[1]
    assert "from_unixtime(unix_timestamp(current_timestamp()) - 600)" in fake.statements[1]


def test_update_pipeline_run_status_merge_scopes_onboard_rows(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.update_pipeline_run_status("c", "i", "rid", "complete")
    assert "execute_run_id IS NULL" in fake.statements[0]


def test_create_execute_pipeline_run_sql(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.create_execute_pipeline_run("c1", "inst1", "ex-1", "onb-src", db_run_id="j1")
    q = fake.statements[0]
    assert "INSERT INTO" in q
    assert "'ex-1'" in q
    assert "'onb-src'" in q
    assert "'j1'" in q


def test_update_execute_pipeline_run_status(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    pipeline_state.update_execute_pipeline_run_status("c1", "inst1", "ex-9", "complete")
    q = fake.statements[0]
    assert "UPDATE" in q
    assert "execute_run_id" in q
    assert "'ex-9'" in q


def test_bootstrap_execute_run_resumes(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    monkeypatch.setattr(pipeline_state, "create_state_tables", lambda *args, **kwargs: None)
    ex_row = _Row(
        {
            "execute_run_id": "ex-resume",
            "onboard_run_id": "src_ob",
            "status": "running",
        }
    )
    fake.set_sql_result_queue(
        [
            _Result([]),
            _Result([]),
            _Result([ex_row]),
        ]
    )
    out = pipeline_state.bootstrap_execute_run("cat", "inst")
    assert out.execute_run_id == "ex-resume"
    assert out.artifacts_onboard_run_id == "src_ob"


def test_bootstrap_execute_run_raises_without_complete_onboard(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    monkeypatch.setattr(pipeline_state, "create_state_tables", lambda *args, **kwargs: None)
    fake.set_sql_result_queue(
        [
            _Result([]),
            _Result([]),
            _Result([]),
            _Result([]),
        ]
    )
    with pytest.raises(RuntimeError, match="No completed onboard"):
        pipeline_state.bootstrap_execute_run("cat", "inst")


def test_bootstrap_execute_run_mints(monkeypatch) -> None:
    fake = _FakeSpark()
    monkeypatch.setattr(pipeline_state, "get_spark_session", lambda: fake)
    monkeypatch.setattr(pipeline_state, "create_state_tables", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_state, "new_execute_run_id", lambda: "exec-uuid-1")
    onboard_row = _Row(
        {
            "onboard_run_id": "src_onboard",
            "status": "complete",
            "execute_run_id": None,
        }
    )
    fake.set_sql_result_queue(
        [
            _Result([]),
            _Result([]),
            _Result([]),
            _Result([onboard_row]),
            _Result([]),
        ]
    )
    out = pipeline_state.bootstrap_execute_run("cat", "inst")
    assert out.execute_run_id == "exec-uuid-1"
    assert out.artifacts_onboard_run_id == "src_onboard"
