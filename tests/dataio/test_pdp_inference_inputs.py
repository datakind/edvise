"""PDP inference reuses gcs_uploads/{batch_id} and backfills it once."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from edvise.dataio import pdp_inference_inputs as m
from edvise.utils.gcs import SUCCESS_FILENAME


def test_pdp_batch_inputs_ready_requires_both_files_and_marker(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "cohort.csv").write_text("a\n", encoding="utf-8")
    (batch / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
    assert m.pdp_batch_inputs_ready(str(batch), "cohort.csv", "course.csv") is False

    (batch / "course.csv").write_text("b\n", encoding="utf-8")
    assert m.pdp_batch_inputs_ready(str(batch), "cohort.csv", "course.csv") is True


def test_materialize_reuses_ready_batch_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "gcs_uploads" / "batch1"
    batch.mkdir(parents=True)
    (batch / "cohort.csv").write_text("a\n", encoding="utf-8")
    (batch / "course.csv").write_text("b\n", encoding="utf-8")
    (batch / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "bronze_gcs_batch_dir", lambda *_a, **_k: str(batch))
    copy = mock.Mock()
    monkeypatch.setattr(m, "copy_validated_blobs_to_landing", copy)

    course, cohort, landing = m.materialize_pdp_batch_inputs(
        db_workspace="dev_sst_02",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        cohort_file_name="cohort.csv",
        course_file_name="course.csv",
    )

    copy.assert_not_called()
    assert landing == str(batch)
    assert course == str(batch / "course.csv")
    assert cohort == str(batch / "cohort.csv")


def test_materialize_backfills_missing_batch_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = tmp_path / "gcs_uploads" / "batch1"
    monkeypatch.setattr(m, "bronze_gcs_batch_dir", lambda *_a, **_k: str(batch))

    def _copy(**kwargs: object) -> int:
        landing = Path(str(kwargs["landing_dir"]))
        landing.mkdir(parents=True, exist_ok=True)
        (landing / "cohort.csv").write_text("a\n", encoding="utf-8")
        (landing / "course.csv").write_text("b\n", encoding="utf-8")
        return 2

    monkeypatch.setattr(m, "copy_validated_blobs_to_landing", _copy)

    course, cohort, landing = m.materialize_pdp_batch_inputs(
        db_workspace="dev_sst_02",
        databricks_institution_name="school",
        gcp_bucket_name="bucket",
        batch_id="batch1",
        cohort_file_name="cohort.csv",
        course_file_name="course.csv",
    )

    assert landing == str(batch)
    assert Path(course).is_file()
    assert Path(cohort).is_file()
    assert (batch / SUCCESS_FILENAME).is_file()
    assert m.pdp_batch_inputs_ready(str(batch), "cohort.csv", "course.csv") is True
