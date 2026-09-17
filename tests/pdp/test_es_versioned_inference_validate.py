"""Tests for ES versioned-inference validate helpers (PDP-safe)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edvise.runtime.versioned_inference.cli import (  # noqa: E402
    add_es_inference_trigger_args,
    build_es_launcher_parameter_overrides,
    build_es_launcher_trigger_inputs,
    parse_is_genai_institution,
)
from edvise.runtime.versioned_inference.dab_layout import (  # noqa: E402
    genai_execute_dab_bundle_layout,
    genai_snapshot_dir,
    resolve_dab_bundle_layout,
)
from edvise.runtime.versioned_inference.submit import DEFAULT_GIT_URL  # noqa: E402


def _es_args(**overrides: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_es_inference_trigger_args(parser)
    defaults = {
        "databricks_institution_name": "city_cols_of_chicago",
        "model_name": "demo_model",
        "DB_workspace": "dev_sst_02",
        "schema_type": "edvise",
        "release_base_path": "",
        "git_url": "",
        "is_genai_institution": "false",
        "batch_id": "batch-1",
        "validated_blob_paths_json": "[]",
        "config_file_name": "config.toml",
        "genai_inputs_toml_path": "inputs.toml",
        "term_filter": "",
        "gcp_bucket_name": "bucket",
        "datakind_notification_email": "a@datakind.org",
        "DK_CC_EMAIL": "",
        "ds_run_as": "sp",
        "service_account_executer": "sa",
        "datakind_group_to_manage_workflow": "grp",
        "inference_parameters_json": "",
        "launcher_run_id": "12345",
        "model_run_id": "",
        "cohort_file_name": "",
        "course_file_name": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_parse_is_genai_institution() -> None:
    assert parse_is_genai_institution("true") is True
    assert parse_is_genai_institution("FALSE") is False
    assert parse_is_genai_institution("") is False


def test_build_es_launcher_parameter_overrides_includes_es_fields() -> None:
    overrides = build_es_launcher_parameter_overrides(
        _es_args(is_genai_institution="true", batch_id="b1")
    )
    assert overrides["schema_type"] == "edvise"
    assert overrides["is_genai_institution"] == "true"
    assert overrides["batch_id"] == "b1"
    assert overrides["db_run_id"] == "12345"
    assert "cohort_file_name" not in overrides


def test_build_es_launcher_trigger_inputs_uses_es_release_base() -> None:
    inputs = build_es_launcher_trigger_inputs(
        _es_args(), default_git_url=DEFAULT_GIT_URL
    )
    assert inputs.schema_type == "edvise"
    assert inputs.release_base_path.endswith("/edvise_releases/es")
    assert inputs.param_overrides["is_genai_institution"] == "false"


def test_validate_es_requires_genai_snapshot_when_flag_true(
    tmp_path: Path,
) -> None:
    """GenAI path fails fast when materialize did not create …/genai/."""
    from edvise.runtime.versioned_inference.tasks import validate_es

    es_layout = resolve_dab_bundle_layout("edvise")
    release_dir = tmp_path / "es_sha"
    (release_dir / "databricks_bundle_snapshot" / "resources").mkdir(parents=True)
    es_yml = release_dir / es_layout.inference_yml_snapshot_rel
    es_yml.write_text(
        "resources:\n  jobs:\n    github_sourced_genai_es_inference_pipeline:\n"
        "      name: es\n      tasks: []\n      job_clusters:\n"
        "        - job_cluster_key: c\n          new_cluster:\n"
        "            spark_version: 15.4.x-cpu-ml-scala2.12\n"
        "      parameters:\n        - name: DB_workspace\n          default: x\n",
        encoding="utf-8",
    )
    (release_dir / "databricks_bundle_snapshot" / "databricks.yml").write_text(
        "variables:\n  DB_workspace:\n    default: dev_sst_02\n",
        encoding="utf-8",
    )

    spark = MagicMock()
    with (
        patch.object(validate_es, "get_spark_session", return_value=spark),
        patch.object(
            validate_es,
            "resolve_model_run_and_pipeline_version",
            return_value=("run1", "es_sha"),
        ),
        patch.object(validate_es, "record_versioned_inference_launcher_event"),
        patch.object(
            validate_es,
            "resolve_genai_pipeline_version_from_registry",
            return_value="genai_sha",
        ),
        patch.object(
            validate_es,
            "check_runtime_bundle_compatibility",
            return_value=(True, ""),
        ),
        patch.object(
            validate_es,
            "resolve_versioned_job_parameters",
            return_value={},
        ),
        patch.object(validate_es, "record_launcher_failures") as fail_ctx,
    ):
        # Make the context manager a no-op that still exposes event attrs.
        event = MagicMock()
        fail_ctx.return_value.__enter__.return_value = event
        fail_ctx.return_value.__exit__.return_value = None

        argv = [
            "--databricks_institution_name",
            "city_cols_of_chicago",
            "--model_name",
            "demo_model",
            "--DB_workspace",
            "dev_sst_02",
            "--schema_type",
            "edvise",
            "--release_base_path",
            str(tmp_path),
            "--is_genai_institution",
            "true",
            "--launcher_run_id",
            "99",
        ]
        with pytest.raises(FileNotFoundError, match="GenAI release bundle"):
            validate_es.main(argv)


def test_genai_layout_job_key_for_validate() -> None:
    layout = genai_execute_dab_bundle_layout()
    assert layout.inference_job_key == "edvise_genai_mapping_execute_pipeline"
    assert genai_snapshot_dir(Path("/x/es_sha")).name == "genai"
