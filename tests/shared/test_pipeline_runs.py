"""Unit tests for edvise.shared.dashboard_metadata.pipeline_runs."""

import json
from datetime import datetime


import edvise.shared.dashboard_metadata.pipeline_runs as pipeline_runs


class _FakeWriter:
    def __init__(self) -> None:
        self.format_value = None
        self.mode_value = None
        self.options: dict[str, str] = {}
        self.saved_table = None

    def format(self, fmt: str):
        self.format_value = fmt
        return self

    def mode(self, mode: str):
        self.mode_value = mode
        return self

    def option(self, key: str, value: str):
        self.options[key] = value
        return self

    def saveAsTable(self, table_path: str) -> None:
        self.saved_table = table_path


class _FakeDF:
    def __init__(self, *, writer: _FakeWriter) -> None:
        self.write = writer


class _FakeSpark:
    def __init__(self) -> None:
        self.created_rows = None
        self.writer = _FakeWriter()

    def createDataFrame(self, rows, schema=None):
        self.created_rows = rows
        return _FakeDF(writer=self.writer)


def test_parse_ts14_from_filename_none_returns_none():
    assert pipeline_runs.parse_timestamp_from_filename(None) is None


def test_parse_ts14_from_filename_no_ts_returns_none():
    assert pipeline_runs.parse_timestamp_from_filename("cohort.csv") is None
    assert pipeline_runs.parse_timestamp_from_filename("cohort_2025.csv") is None


def test_parse_ts14_from_filename_extracts_ts():
    ts = pipeline_runs.parse_timestamp_from_filename("cohort_20250723040724.csv")
    assert ts == datetime(2025, 7, 23, 4, 7, 24)


def test_append_pipeline_run_event_missing_run_id_returns_false(monkeypatch):
    def _boom():
        raise AssertionError(
            "_get_spark_session should not be called when run_id is empty"
        )

    monkeypatch.setattr(pipeline_runs, "_get_spark_session", _boom)
    ok = pipeline_runs.append_pipeline_run_event(
        catalog="test",
        run_id=None,
        run_type="training",
        event="started",
    )
    assert ok is False


def test_append_pipeline_run_event_missing_catalog_returns_false(monkeypatch):
    class _SparkNoUse:
        def createDataFrame(self, rows):
            raise AssertionError(
                "createDataFrame should not be called when catalog is empty"
            )

    monkeypatch.setattr(pipeline_runs, "_get_spark_session", lambda: _SparkNoUse())
    ok = pipeline_runs.append_pipeline_run_event(
        catalog=None,
        run_id="r1",
        run_type="training",
        event="started",
    )
    assert ok is False


def test_append_pipeline_run_event_success_writes_to_uc_table(monkeypatch):
    fake_spark = _FakeSpark()
    monkeypatch.setattr(pipeline_runs, "_get_spark_session", lambda: fake_spark)

    ok = pipeline_runs.append_pipeline_run_event(
        catalog="my_catalog",
        run_id="r1",
        run_type="training",
        event="completed",
        institution_id="motlow_state_cc",
        databricks_institution_name="motlow_state_cc",
        cohort_dataset_name="cohort_20250723040724.csv",
        course_dataset_name="course_20250723040724.csv",
        dataset_ts=datetime(2025, 7, 23, 4, 7, 24),
        payload={"b": "x", "a": 1},
    )

    assert ok is True
    assert fake_spark.writer.format_value == "delta"
    assert fake_spark.writer.mode_value == "append"
    assert fake_spark.writer.options.get("mergeSchema") == "true"
    assert fake_spark.writer.saved_table == "my_catalog.default.pipeline_runs"

    assert fake_spark.created_rows is not None
    assert len(fake_spark.created_rows) == 1
    row = fake_spark.created_rows[0]

    assert row["run_id"] == "r1"
    assert row["run_url"] is None
    assert row["run_type"] == "training"
    assert row["institution_id"] == "motlow_state_cc"
    assert "databricks_institution_name" not in row
    assert row["status"] == "completed"
    assert row["started_at"] is None
    assert isinstance(row["finished_at"], datetime)
    assert row["cohort_dataset_name"] == "cohort_20250723040724.csv"
    assert row["course_dataset_name"] == "course_20250723040724.csv"
    assert row["dataset_ts"] == datetime(2025, 7, 23, 4, 7, 24)
    assert row["cohort"] is None

    assert isinstance(row["updated_at"], datetime)
    assert isinstance(row["payload_json"], str)
    assert json.loads(row["payload_json"]) == {"a": 1, "b": "x"}
