"""Tests for schema_type → DAB bundle path resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edvise.runtime.versioned_inference import dab_layout as dl


@pytest.mark.parametrize(
    ("schema_type", "pipeline_dir", "job_key", "yml_name", "inference_schema"),
    [
        (
            "pdp",
            "pdp",
            "edvise_github_sourced_pdp_inference_pipeline",
            "github_pdp_inference.yml",
            "pdp",
        ),
        (
            "edvise",
            "es",
            "github_sourced_genai_es_inference_pipeline",
            "github_es_inference.yml",
            "edvise",
        ),
        (
            "es",
            "es",
            "github_sourced_genai_es_inference_pipeline",
            "github_es_inference.yml",
            "edvise",
        ),
        (
            "legacy",
            "legacy",
            "edvise_github_sourced_legacy_inference_pipeline",
            "github_legacy_inference.yml",
            "legacy",
        ),
    ],
)
def test_resolve_dab_bundle_layout(
    schema_type: str,
    pipeline_dir: str,
    job_key: str,
    yml_name: str,
    inference_schema: str,
) -> None:
    layout = dl.resolve_dab_bundle_layout(schema_type)
    assert layout.pipeline_dir == pipeline_dir
    assert layout.inference_job_key == job_key
    assert layout.inference_yml_filename == yml_name
    assert layout.inference_schema_type == inference_schema
    assert layout.dab_repo_path == f"pipelines/{pipeline_dir}/databricks.yml"
    assert layout.inference_yml_repo_path == (
        f"pipelines/{pipeline_dir}/resources/{yml_name}"
    )
    assert layout.inference_yml_snapshot_rel == (
        f"databricks_bundle_snapshot/resources/{yml_name}"
    )


def test_unknown_schema_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown schema_type"):
        dl.resolve_dab_bundle_layout("unknown")
