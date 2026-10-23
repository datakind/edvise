"""NSC webapp batch registration decisions."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from edvise.ingestion.nsc_sftp.webapp_batch import (
    InstitutionPair,
    batch_dir_has_inputs,
    batch_name_for_stamp,
    bucket_from_signed_url,
    external_bucket_name,
    fingerprints_ready_to_close,
    pair_plan_rows,
    plan_registration,
    register_pair,
    split_csv_path,
)
from edvise.utils.gcs import SUCCESS_FILENAME

COHORT = "AO1600_AR_DEIDENTIFIED_STUDYID_20250618030630.csv"
COURSE = "AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20250618030630.csv"
STAMP = "20250618030630"


def _file(name: str, *, valid: bool = True, sst_generated: bool = False) -> dict:
    return {"name": name, "valid": valid, "sst_generated": sst_generated}


def test_plan_registration_uploads_and_creates_when_nothing_exists() -> None:
    plan = plan_registration(
        {"files": [], "batches": []},
        cohort_file_name=COHORT,
        course_file_name=COURSE,
        batch_name=batch_name_for_stamp(STAMP),
    )
    assert plan.upload_names == (COHORT, COURSE)
    assert plan.create_batch is True
    assert plan.error == ""


def test_plan_registration_reuses_complete_batch() -> None:
    plan = plan_registration(
        {
            "files": [_file(COHORT), _file(COURSE)],
            "batches": [
                {
                    "name": "nsc_20250618030630",
                    "batch_id": "abc",
                    "file_names_to_ids": {COHORT: "1", COURSE: "2"},
                }
            ],
        },
        cohort_file_name=COHORT,
        course_file_name=COURSE,
        batch_name="nsc_20250618030630",
    )
    assert plan.upload_names == ()
    assert plan.create_batch is False
    assert plan.batch_id == "abc"


def test_plan_registration_rejects_batch_with_different_files() -> None:
    plan = plan_registration(
        {
            "files": [_file(COHORT), _file(COURSE)],
            "batches": [
                {
                    "name": "nsc_20250618030630",
                    "batch_id": "abc",
                    "file_names_to_ids": {COHORT: "1"},
                }
            ],
        },
        cohort_file_name=COHORT,
        course_file_name=COURSE,
        batch_name="nsc_20250618030630",
    )
    assert "already exists" in plan.error
    assert plan.create_batch is False


def test_plan_registration_rejects_invalid_existing_file() -> None:
    plan = plan_registration(
        {"files": [_file(COHORT, valid=False)], "batches": []},
        cohort_file_name=COHORT,
        course_file_name=COURSE,
        batch_name="nsc_20250618030630",
    )
    assert "not a valid input file" in plan.error


def test_pair_plan_rows_and_split_path() -> None:
    rows = [
        {
            "file_fingerprint": "fp-cohort",
            "file_name": COHORT,
            "institution_id": "1600",
            "inst_id": "insthex",
            "institution_name": "Miles CC",
        },
        {
            "file_fingerprint": "fp-course",
            "file_name": COURSE,
            "institution_id": "1600",
            "inst_id": "insthex",
            "institution_name": "Miles CC",
        },
    ]
    pairs, problems = pair_plan_rows(rows, tmp_dir="/tmp/nsc")
    assert problems == []
    assert len(pairs) == 1
    assert pairs[0].stamp == STAMP
    assert pairs[0].cohort_split_path == split_csv_path(
        "/tmp/nsc", "fp-cohort", "1600", COHORT
    )


def test_fingerprints_ready_ignores_missing_splits() -> None:
    rows = [
        {
            "file_fingerprint": "fp-cohort",
            "inst_id": "inst",
            "stamp": STAMP,
            "split_exists": True,
        },
        {
            "file_fingerprint": "fp-cohort",
            "inst_id": "empty-inst",
            "stamp": STAMP,
            "split_exists": False,
        },
    ]
    assert fingerprints_ready_to_close(rows, {("inst", STAMP, "fp-cohort")}) == [
        "fp-cohort"
    ]
    assert fingerprints_ready_to_close(rows, set()) == []


def test_external_bucket_and_signed_url() -> None:
    assert external_bucket_name("dev_sst_02", "ABC") == "dev_abc"
    url = "https://storage.googleapis.com/dev_abc/unvalidated/file.csv?X-Goog-Algorithm=GOOG4"
    assert bucket_from_signed_url(url) == "dev_abc"


def test_register_pair_reuses_batch_and_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "cohort.csv"
    course = tmp_path / "course.csv"
    cohort.write_text("a\n", encoding="utf-8")
    course.write_text("b\n", encoding="utf-8")
    pair = InstitutionPair(
        inst_id="abc",
        pdp_id="1600",
        institution_name="Miles CC",
        stamp=STAMP,
        cohort_file_name=COHORT,
        course_file_name=COURSE,
        cohort_fingerprint="fp-cohort",
        course_fingerprint="fp-course",
        cohort_split_path=str(cohort),
        course_split_path=str(course),
    )
    monkeypatch.setattr(
        "edvise.utils.api_requests.get_institution_input",
        lambda *_a, **_k: {"files": [], "batches": []},
    )
    monkeypatch.setattr(
        "edvise.utils.api_requests.get_upload_url",
        lambda *_a, **_k: "https://storage.googleapis.com/dev_abc/unvalidated/x.csv",
    )
    monkeypatch.setattr(
        "edvise.utils.api_requests.put_csv_bytes", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "edvise.utils.api_requests.validate_sftp_upload", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        "edvise.utils.api_requests.create_institution_batch",
        lambda *_a, **_k: {
            "batch_id": "batch123",
            "file_names_to_ids": {COHORT: "1", COURSE: "2"},
        },
    )

    def _copy(**kwargs: object) -> str:
        landing = Path(str(kwargs["batch_dir"]))
        landing.mkdir(parents=True)
        (landing / COHORT).write_text("a\n", encoding="utf-8")
        (landing / COURSE).write_text("b\n", encoding="utf-8")
        (landing / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")
        return str(landing)

    monkeypatch.setattr(
        "edvise.ingestion.nsc_sftp.webapp_batch.copy_validated_batch_inputs", _copy
    )
    result = register_pair(
        mock.Mock(),
        pair,
        db_workspace="dev_sst_02",
        bronze_dir=str(tmp_path / "bronze"),
    )
    assert result["batch_id"] == "batch123"
    assert result["status"] == "BRONZE_MATERIALIZED"
    assert batch_dir_has_inputs(result["bronze_path"], COHORT, COURSE)
