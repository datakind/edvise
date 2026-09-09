"""Contract tests for the live PDP inference DAB job YAML."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edvise.runtime.versioned_inference.bundle.from_dab import (
    load_inference_job_definition,
)

_PDP_INFERENCE_YML = _REPO_ROOT / "pipelines/pdp/resources/github_pdp_inference.yml"


def test_pdp_inference_data_audit_receives_ingestion_dataset_paths() -> None:
    """data_audit must receive the CSVs ingestion landed, not only bronze_volume_path."""
    job = load_inference_job_definition(_PDP_INFERENCE_YML)
    data_audit = next(t for t in job["tasks"] if t["task_key"] == "data_audit")
    params = data_audit["spark_python_task"]["parameters"]

    cohort_idx = params.index("--cohort_dataset_validated_path")
    course_idx = params.index("--course_dataset_validated_path")
    assert (
        params[cohort_idx + 1]
        == "{{tasks.data_ingestion.values.cohort_dataset_validated_path}}"
    )
    assert (
        params[course_idx + 1]
        == "{{tasks.data_ingestion.values.course_dataset_validated_path}}"
    )
