"""Tests for ES versioned-inference trigger / dual-pin submit (PDP-safe)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edvise.runtime.versioned_inference.child_run_values import (  # noqa: E402
    DRY_RUN_INGESTION_HANDOFF,
    extract_task_values_from_run,
    require_ingestion_handoff,
)
from edvise.runtime.versioned_inference.es_segments import (  # noqa: E402
    es_full_task_keys,
    es_prefix_task_keys,
    es_suffix_task_keys,
    replace_task_value_refs,
)
from edvise.runtime.versioned_inference.submit_es import (  # noqa: E402
    EsChildRunIds,
    build_genai_execute_parameter_overrides,
    plan_es_versioned_submit,
    submit_es_versioned_inference_from_bundle,
)
from edvise.runtime.versioned_inference.io_chain import (  # noqa: E402
    assert_ingestion_outputs_ready,
)

_ES_YML = _REPO_ROOT / "pipelines/es/resources/github_es_inference.yml"
_GENAI_YML = (
    _REPO_ROOT / "pipelines/genai_mapping/resources/github_genai_mapping_execute.yml"
)
_ES_DAB = _REPO_ROOT / "pipelines/es/databricks.yml"
_GENAI_DAB = _REPO_ROOT / "pipelines/genai_mapping/databricks.yml"


_ES_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_GENAI_SHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _minimal_overrides() -> dict[str, str]:
    return {
        "databricks_institution_name": "city_cols_of_chicago",
        "model_name": "demo_model",
        "DB_workspace": "dev_sst_02",
        "schema_type": "edvise",
        "db_run_id": "launcher-run-99",
        "gcp_bucket_name": "bucket",
        "datakind_notification_email": "ops@example.com",
        "config_file_name": "config.toml",
        "ds_run_as": "sp",
        "service_account_executer": "sa",
        "datakind_group_to_manage_workflow": "grp",
        "is_genai_institution": "false",
        "batch_id": "batch-abc",
        "validated_blob_paths_json": "[]",
        "genai_inputs_toml_path": "inputs.toml",
        "job_type": "inference",
        "pipeline_version": _ES_SHA,
    }


def _stage_release_dir(tmp_path: Path, *, with_genai: bool) -> Path:
    release = tmp_path / "edvise_releases" / "es" / _ES_SHA
    snap = release / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    shutil.copy(_ES_YML, snap / "github_es_inference.yml")
    shutil.copy(_ES_DAB, release / "databricks_bundle_snapshot" / "databricks.yml")
    if with_genai:
        genai = release / "genai" / "databricks_bundle_snapshot" / "resources"
        genai.mkdir(parents=True)
        shutil.copy(_GENAI_YML, genai / "github_genai_mapping_execute.yml")
        shutil.copy(
            _GENAI_DAB,
            release / "genai" / "databricks_bundle_snapshot" / "databricks.yml",
        )
    return release


@pytest.mark.skipif(not _ES_YML.is_file(), reason="ES inference YAML missing")
def test_es_segment_keys_es_full_excludes_condition_and_run_job() -> None:
    raw = yaml.safe_load(_ES_YML.read_text(encoding="utf-8"))
    job = raw["resources"]["jobs"]["github_sourced_genai_es_inference_pipeline"]
    keys = es_full_task_keys(job["tasks"])
    assert "data_ingestion" in keys
    assert "data_audit" in keys
    assert "output_publish" in keys
    assert "check_is_genai_institution" not in keys
    assert "genai_mapping_execute" not in keys
    assert es_prefix_task_keys() == {"data_ingestion"}
    suffix = es_suffix_task_keys(job["tasks"])
    assert "data_ingestion" not in suffix
    assert "data_audit" in suffix
    assert "output_publish" in suffix


def test_es_child_run_ids_payload_names_by_path() -> None:
    full = EsChildRunIds(
        is_genai=False, es_full=111, es_full_url="https://example/full"
    )
    assert full.as_payload() == {
        "is_genai_institution": "false",
        "child_run_es_full": "111",
        "child_inference_run_id": "111",
    }
    genai = EsChildRunIds(
        is_genai=True,
        es_prefix=1,
        genai_execute=2,
        es_suffix=3,
        es_prefix_url="https://example/p",
        genai_execute_url="https://example/g",
        es_suffix_url="https://example/s",
    )
    payload = genai.as_payload()
    assert payload["child_run_es_prefix"] == "1"
    assert payload["child_run_genai_execute"] == "2"
    assert payload["child_run_es_suffix"] == "3"
    assert payload["child_inference_run_id"] == "3"
    assert "child_run_es_prefix_or_classical" not in payload
    assert "child_run_es_full" not in payload
    assert not any(k.endswith("_url") for k in payload)


def test_replace_task_value_refs_only_data_ingestion() -> None:
    text = "{{tasks.data_ingestion.values.config_file_path}}|{{tasks.other.values.x}}"
    out = replace_task_value_refs(text, {"config_file_path": "/Volumes/x/config.toml"})
    assert out == "/Volumes/x/config.toml|{{tasks.other.values.x}}"


def test_require_ingestion_handoff_hard_requires_config() -> None:
    with pytest.raises(ValueError, match="config_file_path"):
        require_ingestion_handoff(
            {"bronze_batch_dir": "/b"},
            required_keys=("bronze_batch_dir", "config_file_path"),
        )


def test_extract_task_values_from_run_dict_shape() -> None:
    run = {
        "tasks": [
            {
                "task_key": "data_ingestion",
                "values": {
                    "bronze_batch_dir": "/Volumes/b/batch",
                    "config_file_path": "/Volumes/c/config.toml",
                },
            }
        ]
    }
    values = extract_task_values_from_run(run, task_key="data_ingestion")
    assert values["bronze_batch_dir"] == "/Volumes/b/batch"
    assert values["config_file_path"] == "/Volumes/c/config.toml"


def test_build_genai_execute_parameter_overrides_suffixes_db_run_id() -> None:
    overrides = build_genai_execute_parameter_overrides(
        {
            "databricks_institution_name": "inst",
            "DB_workspace": "dev_sst_02",
            "db_run_id": "99",
            "genai_inputs_toml_path": "inputs.toml",
        },
        genai_pipeline_version="genai_sha",
        bronze_batch_dir="/Volumes/b/batch",
    )
    assert overrides["institution_id"] == "inst"
    assert overrides["catalog"] == "dev_sst_02"
    assert overrides["db_run_id"] == "99_genai_execute"
    assert overrides["bronze_batch_dir"] == "/Volumes/b/batch"
    assert overrides["pipeline_version"] == "genai_sha"


@pytest.mark.skipif(
    not (_ES_YML.is_file() and _ES_DAB.is_file()),
    reason="ES pipeline files missing",
)
def test_plan_es_full_submit_strips_clusterless_tasks(tmp_path: Path) -> None:
    release = _stage_release_dir(tmp_path, with_genai=False)
    plan = plan_es_versioned_submit(
        release,
        es_pipeline_version=_ES_SHA,
        is_genai=False,
        parameter_overrides=_minimal_overrides(),
    )
    assert plan.es_full_body is not None
    keys = [t["task_key"] for t in plan.es_full_body["tasks"]]
    assert "data_ingestion" in keys
    assert "check_is_genai_institution" not in keys
    assert "genai_mapping_execute" not in keys
    assert plan.es_full_body["git_source"]["git_commit"] == _ES_SHA
    # ES-full keeps task-value refs inside one run (Databricks resolves them).
    assert all("new_cluster" in t for t in plan.es_full_body["tasks"])


@pytest.mark.skipif(
    not (_ES_YML.is_file() and _GENAI_YML.is_file() and _ES_DAB.is_file()),
    reason="ES/GenAI pipeline files missing",
)
def test_plan_genai_dual_pin_three_bodies(tmp_path: Path) -> None:
    release = _stage_release_dir(tmp_path, with_genai=True)
    handoff = dict(DRY_RUN_INGESTION_HANDOFF)
    plan = plan_es_versioned_submit(
        release,
        es_pipeline_version=_ES_SHA,
        is_genai=True,
        parameter_overrides={**_minimal_overrides(), "is_genai_institution": "true"},
        genai_pipeline_version=_GENAI_SHA,
        handoff=handoff,
    )
    assert plan.prefix_body is not None
    assert plan.genai_body is not None
    assert plan.suffix_body is not None
    prefix_keys = [t["task_key"] for t in plan.prefix_body["tasks"]]
    assert prefix_keys == ["data_ingestion"]
    genai_keys = [t["task_key"] for t in plan.genai_body["tasks"]]
    assert genai_keys == ["ia_execute", "sma_execute"]
    assert plan.genai_body["git_source"]["git_commit"] == _GENAI_SHA
    assert plan.prefix_body["git_source"]["git_commit"] == _ES_SHA
    assert plan.suffix_body["git_source"]["git_commit"] == _ES_SHA
    suffix_blob = str(plan.suffix_body)
    assert "{{tasks.data_ingestion.values" not in suffix_blob
    assert handoff["config_file_path"] in suffix_blob


@pytest.mark.skipif(
    not (_ES_YML.is_file() and _GENAI_YML.is_file()),
    reason="ES/GenAI pipeline files missing",
)
def test_submit_es_genai_dry_run_orchestrates_three_phases(tmp_path: Path) -> None:
    release = _stage_release_dir(tmp_path, with_genai=True)
    with patch(
        "edvise.runtime.versioned_inference.submit_es.resolve_genai_pipeline_version_from_registry",
        return_value=_GENAI_SHA,
    ):
        ids = submit_es_versioned_inference_from_bundle(
            release,
            es_pipeline_version=_ES_SHA,
            is_genai=True,
            parameter_overrides={
                **_minimal_overrides(),
                "is_genai_institution": "true",
            },
            dry_run=True,
            wait_for_completion=False,
            db_workspace="dev_sst_02",
            databricks_institution_name="city_cols_of_chicago",
        )
    assert ids.es_prefix == 0
    assert ids.genai_execute == 0
    assert ids.es_suffix == 0
    assert ids.is_genai is True
    assert "child_run_es_prefix" in ids.as_payload()


@pytest.mark.skipif(not _ES_YML.is_file(), reason="ES inference YAML missing")
def test_submit_es_full_dry_run(tmp_path: Path) -> None:
    release = _stage_release_dir(tmp_path, with_genai=False)
    ids = submit_es_versioned_inference_from_bundle(
        release,
        es_pipeline_version=_ES_SHA,
        is_genai=False,
        parameter_overrides=_minimal_overrides(),
        dry_run=True,
        wait_for_completion=False,
    )
    assert ids.es_full == 0
    assert ids.es_prefix is None
    assert ids.genai_execute is None
    assert ids.es_suffix is None
    assert ids.as_payload()["child_run_es_full"] == "0"


def test_assert_ingestion_outputs_ready_requires_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("x=1\n", encoding="utf-8")
    assert_ingestion_outputs_ready(
        {"config_file_path": str(config), "bronze_batch_dir": ""},
    )
    with pytest.raises(FileNotFoundError, match="config_file_path"):
        assert_ingestion_outputs_ready(
            {
                "config_file_path": str(tmp_path / "missing.toml"),
                "bronze_batch_dir": "",
            },
        )


def test_submit_es_genai_waits_prefix_then_handoff(tmp_path: Path) -> None:
    release = _stage_release_dir(tmp_path, with_genai=True)
    client = MagicMock()
    submit_calls: list[dict] = []

    def fake_submit(body, *, dry_run=False, workspace_client=None, logger=None):
        submit_calls.append(body)
        return 100 + len(submit_calls)

    with (
        patch(
            "edvise.runtime.versioned_inference.submit_es.resolve_genai_pipeline_version_from_registry",
            return_value=_GENAI_SHA,
        ),
        patch(
            "edvise.runtime.versioned_inference.submit_es.submit_inference_run",
            side_effect=fake_submit,
        ),
        patch(
            "edvise.runtime.versioned_inference.submit_es.wait_for_inference_run",
        ) as wait_mock,
        patch(
            "edvise.runtime.versioned_inference.submit_es.resolve_handoff_after_prefix",
            return_value=dict(DRY_RUN_INGESTION_HANDOFF),
        ),
        patch(
            "edvise.runtime.versioned_inference.submit_es.assert_ingestion_outputs_ready",
        ),
        patch(
            "edvise.runtime.versioned_inference.submit_es.assert_genai_execute_outputs_ready",
            return_value="exec_1",
        ),
        patch(
            "edvise.runtime.versioned_inference.submit_es.assert_es_inference_outputs_ready",
        ),
        patch(
            "edvise.runtime.versioned_inference.submit_es.fetch_run_page_url",
            return_value="https://example.databricks.com/#job/1/run/9",
        ),
    ):
        ids = submit_es_versioned_inference_from_bundle(
            release,
            es_pipeline_version=_ES_SHA,
            is_genai=True,
            parameter_overrides={
                **_minimal_overrides(),
                "is_genai_institution": "true",
            },
            dry_run=False,
            wait_for_completion=True,
            workspace_client=client,
            db_workspace="dev_sst_02",
            databricks_institution_name="city_cols_of_chicago",
            model_run_id="modelrun123",
        )
    assert len(submit_calls) == 3
    assert [t["task_key"] for t in submit_calls[0]["tasks"]] == ["data_ingestion"]
    assert [t["task_key"] for t in submit_calls[1]["tasks"]] == [
        "ia_execute",
        "sma_execute",
    ]
    assert "data_audit" in [t["task_key"] for t in submit_calls[2]["tasks"]]
    assert wait_mock.call_count == 3
    assert ids.es_prefix == 101
    assert ids.genai_execute == 102
    assert ids.es_suffix == 103
    assert ids.primary == 103
    assert ids.as_payload()["child_run_es_prefix"] == "101"
    assert ids.es_prefix_url is not None
