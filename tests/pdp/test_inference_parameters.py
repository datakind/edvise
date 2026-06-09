"""Tests for versioned inference parameter contract resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.bundle_from_dab import load_inference_job_definition
from pipelines.pdp.launchers.inference_parameters import (
    build_parameter_contract,
    build_stable_trigger_payload,
    load_parameter_aliases,
    resolve_archived_parameter_values,
    resolve_versioned_job_parameters,
)

_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "inference_job_parameter_contract.yml"
)


def test_build_parameter_contract_marks_referenced_required() -> None:
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    by_name = {spec.name: spec for spec in contract}

    assert by_name["schema_type"].required is True
    assert "data_ingestion" in by_name["schema_type"].referenced_by_tasks
    assert by_name["job_type"].required is False
    assert by_name["job_type"].default == "inference"
    assert by_name["databricks_institution_name"].required is False


def test_resolve_archived_values_literal_default_and_overrides() -> None:
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    values = resolve_archived_parameter_values(
        contract,
        launcher_overrides={
            "databricks_institution_name": "synthetic_integration",
            "cohort_file_name": "cohort.csv",
            "schema_type": "pdp",
        },
    )
    assert values["databricks_institution_name"] == "synthetic_integration"
    assert values["cohort_file_name"] == "cohort.csv"
    assert values["schema_type"] == "pdp"
    assert values["job_type"] == "inference"


def test_resolve_archived_values_fails_when_required_missing() -> None:
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    with pytest.raises(ValueError, match="schema_type"):
        resolve_archived_parameter_values(
            contract,
            launcher_overrides={
                "databricks_institution_name": "miles_cc",
                "cohort_file_name": "cohort.csv",
            },
        )


def test_parameter_aliases_rename_launcher_key() -> None:
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    values = resolve_archived_parameter_values(
        contract,
        launcher_overrides={
            "cohort_file_name": "from_launcher.csv",
            "schema_type": "pdp",
        },
        parameter_aliases={"cohort_filename": "cohort_file_name"},
    )
    assert values["cohort_filename"] == "from_launcher.csv"


def test_parameter_aliases_stable_trigger_path(tmp_path: Path) -> None:
    aliases_path = tmp_path / "parameter_aliases.json"
    aliases_path.write_text(
        json.dumps(
            {
                "parameter_aliases": {
                    "cohort_file_name": "datasets.cohort",
                    "cohort_filename": "datasets.cohort",
                    "schema_type": "institution",
                }
            }
        ),
        encoding="utf-8",
    )
    aliases = load_parameter_aliases(tmp_path)
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    stable = build_stable_trigger_payload(
        institution="miles_cc",
        model_name="retention",
        workspace="dev_sst_02",
        cohort_dataset="stable_cohort.csv",
    )
    values = resolve_archived_parameter_values(
        contract,
        launcher_overrides={},
        stable_trigger=stable,
        parameter_aliases=aliases,
    )
    assert values["cohort_file_name"] == "stable_cohort.csv"
    assert values["cohort_filename"] == "stable_cohort.csv"
    assert values["schema_type"] == "miles_cc"


def test_extra_overrides_must_use_archived_names() -> None:
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    with pytest.raises(ValueError, match="unknown archived parameter"):
        resolve_archived_parameter_values(
            contract,
            launcher_overrides={"cohort_file_name": "c.csv"},
            extra_overrides={"not_a_real_param": "x"},
        )


def test_extra_overrides_escape_hatch_for_old_names() -> None:
    job = load_inference_job_definition(_FIXTURE)
    contract = build_parameter_contract(job)
    values = resolve_archived_parameter_values(
        contract,
        launcher_overrides={"cohort_file_name": "c.csv"},
        extra_overrides={"schema_type": "es"},
    )
    assert values["schema_type"] == "es"


def test_resolve_versioned_job_parameters_from_release_dir(tmp_path: Path) -> None:
    snap = tmp_path / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    (snap / "github_pdp_inference.yml").write_text(
        _FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "parameter_aliases.json").write_text(
        json.dumps({"cohort_filename": "cohort_file_name"}),
        encoding="utf-8",
    )
    job = load_inference_job_definition(snap / "github_pdp_inference.yml")
    values = resolve_versioned_job_parameters(
        job,
        tmp_path,
        launcher_overrides={
            "cohort_file_name": "aliased.csv",
            "schema_type": "pdp",
        },
    )
    assert values["cohort_filename"] == "aliased.csv"
