"""Unit tests for edvise.shared.dashboard_metadata.pipeline_models."""

import json
from datetime import datetime, timezone


import edvise.shared.dashboard_metadata.pipeline_models as pipeline_models


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

    def createDataFrame(self, rows):
        self.created_rows = rows
        return _FakeDF(writer=self.writer)


def test_upsert_pipeline_model_missing_required_returns_false(monkeypatch):
    fake_spark = _FakeSpark()
    monkeypatch.setattr(pipeline_models, "_get_spark_session", lambda: fake_spark)

    ok = pipeline_models.upsert_pipeline_model(
        catalog="c",
        institution_id=None,
        model_name="m",
        model_run_id="r",
    )
    assert ok is False


def test_upsert_pipeline_model_writes_row_append_fallback(monkeypatch):
    fake_spark = _FakeSpark()
    monkeypatch.setattr(pipeline_models, "_get_spark_session", lambda: fake_spark)

    # Avoid MLflow/delta dependencies in unit tests
    monkeypatch.setattr(pipeline_models, "_best_effort_fetch_mlflow_run_metrics", lambda **_k: None)
    monkeypatch.setattr(pipeline_models, "_best_effort_resolve_uc_model_version", lambda **_k: None)

    ok = pipeline_models.upsert_pipeline_model(
        catalog="my_catalog",
        institution_id="motlow_state_cc",
        model_name="retention_into_year_2_associates",
        model_run_id="mlflow_run_123",
        training_run_id="db_run_456",
        training_cohort_dataset_name="cohort_20250723040724.csv",
        training_course_dataset_name="course_20250723040724.csv",
        model_card_path="/Volumes/my_catalog/motlow_state_cc_gold/gold_volume/model_cards/mlflow_run_123/model-card-retention_into_year_2_associates.pdf",
        payload={"x": 1},
    )

    assert ok is True
    assert fake_spark.writer.format_value == "delta"
    assert fake_spark.writer.mode_value == "append"
    assert fake_spark.writer.options.get("mergeSchema") == "true"
    assert fake_spark.writer.saved_table == "my_catalog.default.pipeline_models"

    assert fake_spark.created_rows is not None
    assert len(fake_spark.created_rows) == 1
    row = fake_spark.created_rows[0]

    assert isinstance(row["logged_ts"], datetime)
    assert row["logged_ts"].tzinfo == timezone.utc

    assert row["institution_id"] == "motlow_state_cc"
    assert row["model_name"] == "retention_into_year_2_associates"
    assert row["model_version"] is None
    assert row["model_run_id"] == "mlflow_run_123"
    assert row["training_run_id"] == "db_run_456"
    assert row["training_cohort_dataset_name"] == "cohort_20250723040724.csv"
    assert row["training_course_dataset_name"] == "course_20250723040724.csv"
    assert row["model_card_path"].endswith(".pdf")

    assert row["summary_metrics"] is None
    assert row["bias_summary"] is None
    assert json.loads(row["payload_json"]) == {"x": 1}

