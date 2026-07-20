"""Tests for versioned inference Jobs API submit from archived YAML."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.bundle_from_dab import load_inference_job_definition
from pipelines.pdp.launchers.inference_job_submit import (
    _run_state_fields,
    build_submit_access_control_list,
    build_submit_run_body,
    ensure_concrete_db_run_id,
    normalize_versioned_inference_db_run_id,
    propagate_union_libraries_for_submit,
    render_job_parameter_refs,
    resolve_job_parameter_specs,
    submit_versioned_inference_from_bundle,
    wait_for_inference_run,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "inference_job_minimal.yml"
_PARAM_FIXTURE = (
    Path(__file__).parent / "fixtures" / "inference_job_parameter_contract.yml"
)
_FULL_YML = (
    _REPO_ROOT
    / "pipelines/pdp/examples/versioned_runtime_bundle.example"
    / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
)


def test_resolve_job_parameter_specs_strips_bundle_vars() -> None:
    specs = [
        {
            "name": "databricks_institution_name",
            "default": "${var.databricks_institution_name}",
        },
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


def test_render_job_parameter_refs_nested() -> None:
    rendered = render_job_parameter_refs(
        {
            "parameters": [
                "--cohort_file_name",
                "{{job.parameters.cohort_file_name}}",
                "/Volumes/{{job.parameters.DB_workspace}}/{{job.parameters.databricks_institution_name}}_bronze",
            ]
        },
        {
            "cohort_file_name": "cohort.csv",
            "DB_workspace": "dev_sst_02",
            "databricks_institution_name": "miles_cc",
        },
        run_id="run-abc",
    )
    assert rendered["parameters"][1] == "cohort.csv"
    assert rendered["parameters"][2] == "/Volumes/dev_sst_02/miles_cc_bronze"


def test_render_job_parameter_refs_missing_raises() -> None:
    with pytest.raises(ValueError, match="cohort_file_name"):
        render_job_parameter_refs("{{job.parameters.cohort_file_name}}", {})


def test_ensure_concrete_db_run_id_requires_parent_or_override() -> None:
    with pytest.raises(ValueError, match="launcher_run_id"):
        ensure_concrete_db_run_id({"db_run_id": "{{job.run_id}}"}, {})


def test_ensure_concrete_db_run_id_uses_parent_launcher_run_id() -> None:
    rid = ensure_concrete_db_run_id(
        {"db_run_id": "{{job.run_id}}"},
        {"db_run_id": "439619245566927"},
    )
    assert rid == "439619245566927"


def test_normalize_versioned_inference_db_run_id_bare_hex() -> None:
    assert (
        normalize_versioned_inference_db_run_id("67B39F0B2CAA4A919F289749883BD04C")
        == "versioned_67b39f0b2caa4a919f289749883bd04c"
    )


def test_normalize_versioned_inference_db_run_id_dashed_uuid() -> None:
    assert (
        normalize_versioned_inference_db_run_id("67b39f0b-2caa-4a91-9f28-9749883bd04c")
        == "versioned_67b39f0b2caa4a919f289749883bd04c"
    )


def test_normalize_versioned_inference_db_run_id_already_prefixed() -> None:
    assert (
        normalize_versioned_inference_db_run_id(
            "versioned_67b39f0b2caa4a919f289749883bd04c"
        )
        == "versioned_67b39f0b2caa4a919f289749883bd04c"
    )


def test_normalize_versioned_inference_db_run_id_numeric_parent() -> None:
    assert (
        normalize_versioned_inference_db_run_id("439619245566927") == "439619245566927"
    )


def test_ensure_concrete_db_run_id_respects_hex_override() -> None:
    rid = ensure_concrete_db_run_id(
        {"db_run_id": "{{job.run_id}}"},
        {"db_run_id": "67b39f0b2caa4a919f289749883bd04c"},
    )
    assert rid == "versioned_67b39f0b2caa4a919f289749883bd04c"


def test_build_submit_access_control_list() -> None:
    acl = build_submit_access_control_list(
        {
            "datakind_group_to_manage_workflow": "edvise-admins",
            "viewer_user": "alice@example.com",
        }
    )
    assert acl == [
        {"group_name": "edvise-admins", "permission_level": "CAN_MANAGE_RUN"},
        {"user_name": "alice@example.com", "permission_level": "CAN_VIEW"},
    ]
    assert build_submit_access_control_list({}) == []


def test_build_submit_run_body_includes_acl() -> None:
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
            "datakind_group_to_manage_workflow": "edvise-admins",
            "viewer_user": "bob@example.com",
        },
    )
    assert body["access_control_list"] == [
        {"group_name": "edvise-admins", "permission_level": "CAN_MANAGE_RUN"},
        {"user_name": "bob@example.com", "permission_level": "CAN_VIEW"},
    ]


def test_build_submit_run_body_from_minimal_fixture() -> None:
    job = load_inference_job_definition(_FIXTURE)
    sha = "87b641939205110d03ce8c300e68980327dd6732"
    body = build_submit_run_body(
        job,
        pipeline_version=sha,
        git_url="https://github.com/datakind/edvise",
        run_name="test-run",
        parameter_overrides={
            "databricks_institution_name": "miles_cc",
            "model_name": "retention_into_year_2_associates",
            "DB_workspace": "dev_sst_02",
        },
    )
    assert body["git_source"]["git_commit"] == sha
    assert len(body["tasks"]) == 2
    assert body["tasks"][0]["task_key"] == "feature_generation"
    assert "job_clusters" not in body
    assert "new_cluster" in body["tasks"][0]
    assert "job_cluster_key" not in body["tasks"][0]
    assert "permissions" not in body
    assert "access_control_list" not in body


def test_build_submit_run_body_uses_git_tag_for_release() -> None:
    job = load_inference_job_definition(_FIXTURE)
    body = build_submit_run_body(
        job,
        pipeline_version="v2.4.0",
        git_url="https://github.com/datakind/edvise",
        run_name="test-tag-run",
        parameter_overrides={
            "databricks_institution_name": "miles_cc",
            "model_name": "retention_into_year_2_associates",
            "DB_workspace": "staging_sst_01",
        },
    )
    assert body["git_source"]["git_tag"] == "v2.4.0"
    assert "git_commit" not in body["git_source"]


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
            "db_run_id": "versioned_test_run_001",
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

    body_json = json.dumps(body)
    assert "{{job.parameters" not in body_json
    assert "{{job.run_id}}" not in body_json

    ingestion = next(t for t in body["tasks"] if t["task_key"] == "data_ingestion")
    params = ingestion["spark_python_task"]["parameters"]
    assert "retention_into_year_2_bachelors" in params
    assert "synthetic_integration" in params
    assert "versioned_test_run_001" in params
    assert "{{job.parameters.cohort_file_name}}" not in params

    output_publish = next(t for t in body["tasks"] if t["task_key"] == "output_publish")
    publish_pkgs = {
        lib["pypi"]["package"].split("==")[0].split("~=")[0].lower()
        for lib in output_publish.get("libraries", [])
        if isinstance(lib, dict) and isinstance(lib.get("pypi"), dict)
    }
    assert "pandera" in publish_pkgs
    assert "pydantic" in publish_pkgs


def test_propagate_union_libraries_for_submit() -> None:
    tasks = [
        {
            "task_key": "heavy",
            "libraries": [{"pypi": {"package": "pandera==0.23.0"}}],
        },
        {
            "task_key": "light",
            "libraries": [{"pypi": {"package": "pandas==2.2.3"}}],
        },
    ]
    out = propagate_union_libraries_for_submit(tasks)
    light = next(t for t in out if t["task_key"] == "light")
    names = {
        lib["pypi"]["package"].split("==")[0].lower() for lib in light["libraries"]
    }
    assert "pandera" in names
    assert "pandas" in names


class _FakeRunState:
    def __init__(self, life_cycle_state: str, result_state: str | None = None) -> None:
        self.life_cycle_state = life_cycle_state
        self.result_state = result_state


class _FakeRun:
    def __init__(self, life_cycle_state: str, result_state: str | None = None) -> None:
        self.state = _FakeRunState(life_cycle_state, result_state)
        self.run_page_url = "https://example.com/run/1"


class _FakeJobs:
    def __init__(self, states: list[tuple[str, str | None]]) -> None:
        self._states = states
        self._calls = 0

    def get_run(self, *, run_id: int) -> _FakeRun:
        del run_id
        life, result = self._states[min(self._calls, len(self._states) - 1)]
        self._calls += 1
        return _FakeRun(life, result)


class _FakeWorkspaceClient:
    def __init__(self, states: list[tuple[str, str | None]]) -> None:
        self.jobs = _FakeJobs(states)


def test_run_state_fields_parent_running_all_tasks_success() -> None:
    """Parent run can stay RUNNING while every task is already TERMINATED/SUCCESS."""
    run = {
        "state": {"life_cycle_state": "RUNNING", "result_state": None},
        "tasks": [
            {"state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}},
            {"state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}},
        ],
    }
    assert _run_state_fields(run) == ("TERMINATED", "SUCCESS")


def test_run_state_fields_coerces_sdk_enum_strings() -> None:
    run = {
        "state": {
            "life_cycle_state": "RunLifeCycleState.TERMINATED",
            "result_state": "RunResultState.SUCCESS",
        },
    }
    assert _run_state_fields(run) == ("TERMINATED", "SUCCESS")


def test_wait_for_inference_run_success_when_parent_stays_running() -> None:
    """Simulate multi-task submit: parent RUNNING forever, tasks already done."""

    class _FakeTask:
        def __init__(self, life: str, result: str | None) -> None:
            self.state = _FakeRunState(life, result)

    class _FakeMultiRun:
        def __init__(self) -> None:
            self.state = _FakeRunState("RUNNING", None)
            self.tasks = [
                _FakeTask("TERMINATED", "SUCCESS"),
                _FakeTask("TERMINATED", "SUCCESS"),
            ]
            self.run_page_url = "https://example.com/run/multi"

    class _FakeJobsMulti:
        def get_run(self, *, run_id: int) -> _FakeMultiRun:
            del run_id
            return _FakeMultiRun()

    client = type("C", (), {"jobs": _FakeJobsMulti()})()
    wait_for_inference_run(99, workspace_client=client, poll_interval_seconds=0)


def test_wait_for_inference_run_success() -> None:
    client = _FakeWorkspaceClient(
        [("RUNNING", None), ("TERMINATED", "SUCCESS")],
    )
    wait_for_inference_run(42, workspace_client=client, poll_interval_seconds=0)


def test_wait_for_inference_run_failure() -> None:
    client = _FakeWorkspaceClient([("TERMINATED", "FAILED")])
    with pytest.raises(RuntimeError, match="run_id=42"):
        wait_for_inference_run(42, workspace_client=client, poll_interval_seconds=0)


def test_submit_versioned_inference_validates_required_parameters(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    (snap / "github_pdp_inference.yml").write_text(
        _PARAM_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_type"):
        submit_versioned_inference_from_bundle(
            tmp_path,
            pipeline_version="abc123",
            parameter_overrides={
                "databricks_institution_name": "miles_cc",
                "cohort_file_name": "cohort.csv",
            },
            dry_run=True,
        )


def test_submit_versioned_inference_dry_run_with_resolved_parameters(
    tmp_path: Path,
) -> None:
    snap = tmp_path / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    (snap / "github_pdp_inference.yml").write_text(
        _PARAM_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    run_id = submit_versioned_inference_from_bundle(
        tmp_path,
        pipeline_version="abc123",
        parameter_overrides={
            "databricks_institution_name": "miles_cc",
            "cohort_file_name": "cohort.csv",
            "schema_type": "pdp",
            "viewer_user": "alice@example.com",
            "datakind_group_to_manage_workflow": "edvise-admins",
        },
        dry_run=True,
    )
    assert run_id == 0


def test_submit_versioned_inference_applies_launcher_acl_after_param_resolution(
    tmp_path: Path,
) -> None:
    """Launcher-only ACL keys must survive archived parameter resolution."""
    snap = tmp_path / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    (snap / "github_pdp_inference.yml").write_text(
        _PARAM_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "databricks_bundle_snapshot" / "databricks.yml").write_text(
        "variables:\n  schema_type:\n    default: pdp\n",
        encoding="utf-8",
    )
    job = load_inference_job_definition(snap / "github_pdp_inference.yml")
    archived = {
        "databricks_institution_name": "miles_cc",
        "cohort_file_name": "cohort.csv",
        "schema_type": "pdp",
    }
    body = build_submit_run_body(
        job,
        pipeline_version="abc123",
        git_url="https://github.com/datakind/edvise",
        run_name="test-run",
        parameter_overrides=archived,
        access_control_overrides={
            **archived,
            "viewer_user": "bob@example.com",
            "datakind_group_to_manage_workflow": "edvise-admins",
        },
    )
    assert body["access_control_list"] == [
        {"group_name": "edvise-admins", "permission_level": "CAN_MANAGE_RUN"},
        {"user_name": "bob@example.com", "permission_level": "CAN_VIEW"},
    ]
