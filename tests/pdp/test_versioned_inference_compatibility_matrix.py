"""
Compatibility matrix for versioned inference (features 1–7 acceptance gate).

Validates SHA vs tag git_source, release dir layout, and parameter resolution
across fixture bundles without calling Databricks APIs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.bundle_from_dab import load_inference_job_definition
from pipelines.pdp.launchers.inference_job_submit import build_submit_run_body
from pipelines.pdp.launchers.inference_parameters import (
    resolve_versioned_job_parameters,
)
from pipelines.pdp.launchers.model_metadata import resolve_release_dir
from pipelines.pdp.launchers.pipeline_version_ref import build_git_source

_FIXTURE = Path(__file__).parent / "fixtures" / "inference_job_parameter_contract.yml"

MATRIX = (
    {
        "label": "dev_sha",
        "pipeline_version": "87b641939205110d03ce8c300e68980327dd6732",
        "workspace": "dev_sst_02",
        "release_base": "/Volumes/dev_sst_02/default/edvise_releases",
        "git_key": "git_commit",
    },
    {
        "label": "staging_tag",
        "pipeline_version": "v2.4.0",
        "workspace": "staging_sst_01",
        "release_base": "/Volumes/staging_sst_01/default/edvise_releases",
        "git_key": "git_tag",
    },
)


def _write_bundle(release_dir: Path) -> None:
    snap = release_dir / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "github_pdp_inference.yml").write_text(
        _FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    dab = release_dir / "databricks_bundle_snapshot" / "databricks.yml"
    dab.write_text(
        'variables:\n  schema_type:\n    default: "pdp"\n',
        encoding="utf-8",
    )


@pytest.mark.parametrize("case", MATRIX, ids=[c["label"] for c in MATRIX])
def test_compatibility_matrix_git_source_and_params(
    case: dict[str, str], tmp_path: Path
) -> None:
    pv = case["pipeline_version"]
    release_dir = resolve_release_dir(case["release_base"], pv)
    assert str(release_dir).endswith(pv.replace("/", "_"))
    _write_bundle(tmp_path / release_dir.name)

    git_src = build_git_source("https://github.com/datakind/edvise", pv)
    assert case["git_key"] in git_src
    assert git_src[case["git_key"]] == pv

    job = load_inference_job_definition(
        tmp_path
        / release_dir.name
        / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
    )
    resolved = resolve_versioned_job_parameters(
        job,
        tmp_path / release_dir.name,
        launcher_overrides={
            "databricks_institution_name": "miles_cc",
            "model_name": "retention_into_year_2_associates",
            "DB_workspace": case["workspace"],
            "cohort_file_name": "cohort.csv",
            "course_file_name": "course.csv",
            "gcp_bucket_name": "bucket",
            "datakind_notification_email": "ops@example.com",
        },
        stable_trigger=json.loads(
            json.dumps(
                {
                    "institution": "miles_cc",
                    "model": "retention_into_year_2_associates",
                    "workspace": case["workspace"],
                    "datasets": {"cohort": "cohort.csv", "course": "course.csv"},
                    "outputs": {"bucket": "bucket"},
                    "notifications": {"to": "ops@example.com"},
                }
            )
        ),
    )
    assert resolved["databricks_institution_name"] == "miles_cc"
    assert resolved["cohort_file_name"] == "cohort.csv"

    body = build_submit_run_body(
        job,
        pipeline_version=pv,
        git_url="https://github.com/datakind/edvise",
        run_name=f"matrix-{case['label']}",
        parameter_overrides=resolved,
    )
    assert case["git_key"] in body["git_source"]
    assert body["git_source"][case["git_key"]] == pv
    assert len(body["tasks"]) >= 1
