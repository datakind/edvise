"""Tests for batch GCS inference ingest helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from edvise.dataio import batch_gcs_inference_ingest as m
from edvise.utils.gcs import SUCCESS_FILENAME


def test_bronze_gcs_batch_dir_path() -> None:
    assert m.bronze_gcs_batch_dir("dev_sst_02", "acme", "abc123") == (
        "/Volumes/dev_sst_02/acme_bronze/bronze_volume/gcs_uploads/abc123"
    )


def test_is_bronze_batch_ready_requires_marker_and_data(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    assert m.is_bronze_batch_ready(str(batch)) is False

    (batch / "cohort.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert m.is_bronze_batch_ready(str(batch)) is False

    (batch / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
    assert m.is_bronze_batch_ready(str(batch)) is True


def test_should_skip_batch_ingest() -> None:
    assert not m.should_skip_batch_ingest(
        is_genai_institution="true",
        validated_blob_paths_json='["validated/a.csv"]',
    )
    assert m.should_skip_batch_ingest(
        is_genai_institution="false",
        validated_blob_paths_json="[]",
    )
    assert not m.should_skip_batch_ingest(
        is_genai_institution="false",
        validated_blob_paths_json='["validated/a.csv"]',
    )


def test_resolve_dataset_file_in_batch_dir_exact_and_substring(tmp_path: Path) -> None:
    (tmp_path / "cohort.csv").write_text("x", encoding="utf-8")
    (tmp_path / "other_course.csv").write_text("y", encoding="utf-8")

    assert m.resolve_dataset_file_in_batch_dir(str(tmp_path), "cohort.csv") == str(
        tmp_path / "cohort.csv"
    )
    assert m.resolve_dataset_file_in_batch_dir(str(tmp_path), "course") == str(
        tmp_path / "other_course.csv"
    )


def test_run_batch_gcs_inference_ingest_reuses_ready_bronze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_dir = tmp_path / "gcs_uploads" / "batch1"
    batch_dir.mkdir(parents=True)
    (batch_dir / "cohort.csv").write_text("a\n", encoding="utf-8")
    (batch_dir / "course.csv").write_text("b\n", encoding="utf-8")
    (batch_dir / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        m,
        "bronze_gcs_batch_dir",
        lambda *_a, **_k: str(batch_dir),
    )

    result = m.run_batch_gcs_inference_ingest(
        db_workspace="dev",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        validated_blob_paths_json='["validated/cohort.csv","validated/course.csv"]',
        db_run_id="run-1",
        raw_cohort_name="cohort.csv",
        raw_course_name="course.csv",
        is_genai_institution="false",
    )

    assert result.source == "existing_bronze_batch"
    assert result.copied_count == 0
    assert result.bronze_batch_dir == str(batch_dir)
    assert result.cohort_dataset_validated_path == str(batch_dir / "cohort.csv")
    assert result.course_dataset_validated_path == str(batch_dir / "course.csv")


def test_wait_for_bronze_batch_ready_returns_immediately_when_ready(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "cohort.csv").write_text("a\n", encoding="utf-8")
    (batch / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")

    slept: list[float] = []
    assert m.wait_for_bronze_batch_ready(
        str(batch),
        timeout_seconds=60,
        sleep_fn=slept.append,
    )
    assert slept == []


def test_wait_for_bronze_batch_ready_polls_until_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    ready_after = {"calls": 0}

    def fake_ready(batch_dir: str, *, min_files: int = 1) -> bool:
        ready_after["calls"] += 1
        return ready_after["calls"] >= 2

    monkeypatch.setattr(m, "is_bronze_batch_ready", fake_ready)
    slept: list[float] = []

    assert m.wait_for_bronze_batch_ready(
        str(batch),
        timeout_seconds=30,
        poll_interval_seconds=5,
        sleep_fn=slept.append,
    )
    assert slept == [5.0]


def test_wait_for_bronze_batch_ready_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    monkeypatch.setattr(m, "is_bronze_batch_ready", lambda *_a, **_k: False)

    assert not m.wait_for_bronze_batch_ready(
        str(batch),
        timeout_seconds=15,
        poll_interval_seconds=5,
        sleep_fn=lambda _s: None,
    )


def test_run_batch_gcs_inference_ingest_reuses_after_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_dir = tmp_path / "gcs_uploads" / "batch1"
    batch_dir.mkdir(parents=True)

    def fake_wait(batch_dir_arg: str, **kwargs: object) -> bool:
        (Path(batch_dir_arg) / "cohort.csv").write_text("a\n", encoding="utf-8")
        (Path(batch_dir_arg) / "course.csv").write_text("b\n", encoding="utf-8")
        (Path(batch_dir_arg) / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
        return True

    monkeypatch.setattr(m, "bronze_gcs_batch_dir", lambda *_a, **_k: str(batch_dir))
    monkeypatch.setattr(m, "wait_for_bronze_batch_ready", fake_wait)
    copy = mock.Mock()
    monkeypatch.setattr(m, "copy_validated_blobs_to_landing", copy)

    result = m.run_batch_gcs_inference_ingest(
        db_workspace="dev",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        validated_blob_paths_json='["validated/cohort.csv","validated/course.csv"]',
        db_run_id="run-1",
        raw_cohort_name="cohort.csv",
        raw_course_name="course.csv",
        is_genai_institution="false",
    )

    copy.assert_not_called()
    assert result.source == "existing_bronze_batch"
    assert result.copied_count == 0


def test_run_batch_gcs_inference_ingest_downloads_when_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    landing = tmp_path / "gcs_uploads" / "batch1"
    landing.mkdir(parents=True)

    monkeypatch.setattr(m, "bronze_gcs_batch_dir", lambda *_a, **_k: str(landing))
    monkeypatch.setattr(m, "wait_for_bronze_batch_ready", lambda *_a, **_k: False)

    def fake_copy(**kwargs: object) -> int:
        landing = Path(str(kwargs["landing_dir"]))
        landing.mkdir(parents=True, exist_ok=True)
        (landing / "cohort.csv").write_text("1\n", encoding="utf-8")
        (landing / "course.csv").write_text("2\n", encoding="utf-8")
        return 2

    monkeypatch.setattr(m, "copy_validated_blobs_to_landing", fake_copy)
    monkeypatch.setattr(m, "write_success_marker", lambda *_a, **_k: None)

    result = m.run_batch_gcs_inference_ingest(
        db_workspace="dev",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        validated_blob_paths_json='["validated/cohort.csv","validated/course.csv"]',
        db_run_id="run-1",
        raw_cohort_name="cohort.csv",
        raw_course_name="course.csv",
        is_genai_institution="false",
    )

    assert result.source == "gcs_download"
    assert result.copied_count == 2
    assert result.bronze_batch_dir == str(landing)


def test_run_batch_gcs_inference_ingest_runs_for_genai_with_blob_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    landing = tmp_path / "gcs_uploads" / "batch1"
    landing.mkdir(parents=True)

    monkeypatch.setattr(m, "bronze_gcs_batch_dir", lambda *_a, **_k: str(landing))
    monkeypatch.setattr(m, "wait_for_bronze_batch_ready", lambda *_a, **_k: False)

    def fake_copy(**kwargs: object) -> int:
        landing_dir = Path(str(kwargs["landing_dir"]))
        (landing_dir / "cohort.csv").write_text("1\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(m, "copy_validated_blobs_to_landing", fake_copy)
    monkeypatch.setattr(m, "write_success_marker", lambda *_a, **_k: None)

    result = m.run_batch_gcs_inference_ingest(
        db_workspace="dev",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        validated_blob_paths_json='["validated/cohort.csv"]',
        db_run_id="run-1",
        is_genai_institution="true",
    )

    assert result.skipped is False
    assert result.source == "gcs_download"


def test_run_batch_gcs_inference_ingest_skipped() -> None:
    result = m.run_batch_gcs_inference_ingest(
        db_workspace="dev",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        validated_blob_paths_json="[]",
        db_run_id="run-1",
        is_genai_institution="true",
    )
    assert result.skipped is True
    assert result.bronze_batch_dir == ""


def test_set_batch_ingest_task_values_sets_empty_on_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dbc = mock.Mock()
    monkeypatch.setattr(m, "_get_dbutils", lambda: dbc)

    m.set_batch_ingest_task_values(
        m.BatchIngestResult(
            bronze_batch_dir="",
            cohort_dataset_validated_path=None,
            course_dataset_validated_path=None,
            source="existing_bronze_batch",
            copied_count=0,
            skipped=True,
        )
    )
    dbc.jobs.taskValues.set.assert_any_call(key="bronze_batch_dir", value="")
    dbc.jobs.taskValues.set.assert_any_call(key="ingest_source", value="skipped")
