"""Tests for versioned inference Jobs API submit from archived YAML."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.bundle_from_dab import load_inference_job_definition
from pipelines.pdp.launchers.inference_job_submit import (
    build_submit_run_body,
    resolve_job_parameter_specs,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "inference_job_minimal.yml"
_FULL_YML = (
    _REPO_ROOT
    / "pipelines/pdp/examples/versioned_runtime_bundle.example"
    / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
)


def test_resolve_job_parameter_specs_strips_bundle_vars() -> None:
    specs = [
        {"name": "databricks_institution_name", "default": "${var.databricks_institution_name}"},
        {"name": "DB_workspace", "default": "${var.DB_workspace}"},
        {"name": "db_run_id", "default": "{{job.run_id}}"},
    ]
    out = resolve_job_parameter_specs(
        specs,
        {
            "databricks_institution_name": "miles_cc",
            "DB_workspace": "dev_sst_02",
        },
    )
    by_name = {p["name"]: p["default"] for p in out}
    assert by_name["databricks_institution_name"] == "miles_cc"
    assert by_name["DB_workspace"] == "dev_sst_02"
    assert by_name["db_run_id"] == "{{job.run_id}}"


def test_build_submit_run_body_from_minimal_fixture() -> None:
    job = load_inference_job_definition(_FIXTURE)
    body = build_submit_run_body(
        job,
        pipeline_version="abc123sha456",
        git_url="https://github.com/datakind/edvise",
        run_name="test-run",
        parameter_overrides={
            "databricks_institution_name": "miles_cc",
            "model_name": "retention_into_year_2_associates",
            "DB_workspace": "dev_sst_02",
        },
    )
    assert body["git_source"]["git_commit"] == "abc123sha456"
    assert len(body["tasks"]) == 2
    assert body["tasks"][0]["task_key"] == "feature_generation"
    assert "job_clusters" not in body
    assert "new_cluster" in body["tasks"][0]
    assert "job_cluster_key" not in body["tasks"][0]
    assert "permissions" not in body


@pytest.mark.skipif(not _FULL_YML.is_file(), reason="example bundle snapshot missing")
def test_build_submit_run_body_full_inference_job() -> None:
    job = load_inference_job_definition(_FULL_YML)
    body = build_submit_run_body(
        job,
        pipeline_version="22d2598617be47539a0c478595664e329f234a54",
        git_url="https://github.com/datakind/edvise",
        run_name="versioned-inference-test",
        parameter_overrides={
            "databricks_institution_name": "synthetic_integration",
            "model_name": "retention_into_year_2_bachelors",
            "DB_workspace": "dev_sst_02",
            "datakind_notification_email": "ops@example.com",
        },
    )
    assert len(body["tasks"]) >= 8
    keys = [t["task_key"] for t in body["tasks"]]
    assert "data_ingestion" in keys
    assert "output_publish" in keys
    assert "job_clusters" not in body
    assert all("new_cluster" in t for t in body["tasks"])
    assert body["git_source"]["git_provider"] == "gitHub"
    param_names = {p["name"] for p in body.get("parameters", [])}
    assert "datakind_notification_email" in param_names
    email_default = next(
        p["default"]
        for p in body["parameters"]
        if p["name"] == "datakind_notification_email"
    )
    assert email_default == "ops@example.com"
