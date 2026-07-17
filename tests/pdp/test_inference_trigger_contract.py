"""Tests for edvise-api integration contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers import inference_trigger_contract as itc


def test_build_versioned_inference_job_parameters_flat() -> None:
    params = itc.build_versioned_inference_job_parameters(
        databricks_institution_name="miles_cc",
        model_name="retention_into_year_2_associates",
        db_workspace="dev_sst_02",
        cohort_file_name="cohort.csv",
        course_file_name="course.csv",
        gcp_bucket_name="my-bucket",
        release_base_path="/Volumes/dev_sst_02/default/edvise_releases",
    )
    assert params["databricks_institution_name"] == "miles_cc"
    assert params["model_name"] == "retention_into_year_2_associates"
    assert params["cohort_file_name"] == "cohort.csv"
    assert "stable_trigger_json" not in params
    assert "viewer_user" not in params
    assert "inference_output_run_id" not in params
    assert "inference_parameters_json" not in params


def test_build_versioned_inference_job_parameters_with_extra_json() -> None:
    params = itc.build_versioned_inference_job_parameters(
        databricks_institution_name="miles_cc",
        model_name="retention_into_year_2_associates",
        db_workspace="dev_sst_02",
        inference_parameters_json={
            "db_run_id": "custom-run",
            "DK_CC_EMAIL": "cc@example.com",
        },
    )
    extra = json.loads(params["inference_parameters_json"])
    assert extra["db_run_id"] == "custom-run"
    assert extra["DK_CC_EMAIL"] == "cc@example.com"


def test_job_key_constant() -> None:
    assert (
        itc.VERSIONED_INFERENCE_LAUNCHER_JOB_KEY
        == "edvise_versioned_inference_launcher"
    )
