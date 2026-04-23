"""Tests for :class:`edvise.configs.genai.IdentityAgentInputsConfig`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from edvise.configs.genai import (
    DatasetConfig,
    IdentityAgentInputsConfig,
    SchoolMappingConfig,
    bronze_volume_path_for_institution,
    ia_inputs_toml_under_bronze,
    resolve_genai_data_path,
)
from edvise.genai.mapping.shared.pipeline_artifacts import resolve_pipeline_version
from edvise.dataio.read import from_toml_file


def test_identity_agent_inputs_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "inputs.toml"
    p.write_text(
        textwrap.dedent(
            """
            [institution]
            id = "synthetic_univ_alpha"

            [datasets.files]
            student = "fixture_students.csv"
            course = [
              "fixture_classes_2005_2013.csv",
              "fixture_classes_2014_2025.csv",
            ]
            semester = "fixture_terms.csv"
            """
        ).strip(),
        encoding="utf-8",
    )

    raw = IdentityAgentInputsConfig.model_validate(from_toml_file(str(p)))
    assert raw.institution.id == "synthetic_univ_alpha"
    assert raw.datasets.files["student"] == "fixture_students.csv"
    assert raw.datasets.files["course"] == [
        "fixture_classes_2005_2013.csv",
        "fixture_classes_2014_2025.csv",
    ]

    school = raw.to_school_mapping_config(uc_catalog="dev_sst_02")
    assert school.onboard_run_id is None
    assert school.pipeline_version == resolve_pipeline_version()
    assert school.institution_id == "synthetic_univ_alpha"
    assert school.datasets["student"] == DatasetConfig(
        files=["fixture_students.csv"],
        primary_keys=None,
    )
    assert school.datasets["course"].files == [
        "fixture_classes_2005_2013.csv",
        "fixture_classes_2014_2025.csv",
    ]
    assert school.datasets["course"].primary_keys is None
    assert (
        school.bronze_volumes_path
        == "/Volumes/dev_sst_02/synthetic_univ_alpha_bronze/bronze_volume"
    )


def test_bronze_volume_path_derived_from_institution_id(tmp_path: Path) -> None:
    p = tmp_path / "inputs.toml"
    p.write_text(
        textwrap.dedent(
            """
            [institution]
            id = "synthetic_univ_alpha"

            [datasets.files]
            student = "raw/students.csv"
            """
        ).strip(),
        encoding="utf-8",
    )

    raw = IdentityAgentInputsConfig.model_validate(from_toml_file(str(p)))
    school = raw.to_school_mapping_config(uc_catalog="dev_sst_02")
    assert (
        school.bronze_volumes_path
        == "/Volumes/dev_sst_02/synthetic_univ_alpha_bronze/bronze_volume"
    )
    assert (
        resolve_genai_data_path(
            school.bronze_volumes_path, school.datasets["student"].files[0]
        )
        == "/Volumes/dev_sst_02/synthetic_univ_alpha_bronze/bronze_volume/raw/students.csv"
    )


def test_bronze_volume_path_for_institution_empty_id_raises() -> None:
    with pytest.raises(ValueError, match="institution_id"):
        bronze_volume_path_for_institution("  ", catalog="dev_sst_02")


def test_bronze_volume_path_for_institution_empty_catalog_raises() -> None:
    with pytest.raises(ValueError, match="catalog"):
        bronze_volume_path_for_institution("synthetic_univ_beta", catalog="  ")


def test_bronze_volume_path_for_institution_with_catalog() -> None:
    assert bronze_volume_path_for_institution("synthetic_univ_beta", catalog="my_cat") == (
        "/Volumes/my_cat/synthetic_univ_beta_bronze/bronze_volume"
    )


def test_ia_inputs_toml_under_bronze() -> None:
    assert ia_inputs_toml_under_bronze("synthetic_univ_beta", catalog="my_cat") == (
        "/Volumes/my_cat/synthetic_univ_beta_bronze/bronze_volume/genai_mapping/inputs/inputs.toml"
    )


def test_resolve_genai_data_path_absolute_unchanged() -> None:
    abs_path = "/Volumes/x/file.csv"
    assert resolve_genai_data_path("/Volumes/root", abs_path) == abs_path


def test_resolve_genai_data_path_no_root() -> None:
    assert resolve_genai_data_path(None, "rel/a.csv") == "rel/a.csv"


def test_files_rejects_non_string_list() -> None:
    with pytest.raises(ValidationError):
        IdentityAgentInputsConfig.model_validate(
            {
                "institution": {"id": "x"},
                "datasets": {"files": {"a": [1, 2]}},
            }
        )


def test_dataset_config_rejects_empty_primary_keys_when_set() -> None:
    with pytest.raises(ValueError, match="primary_keys"):
        DatasetConfig(files=["/a.csv"], primary_keys=[])


def test_school_mapping_config_accepts_legacy_pipeline_run_id_toml_key() -> None:
    cfg = SchoolMappingConfig.model_validate(
        {
            "institution_id": "demo",
            "datasets": {"s": {"files": ["a.csv"]}},
            "pipeline_run_id": "from_alias",
        }
    )
    assert cfg.onboard_run_id == "from_alias"


def test_to_school_mapping_config_onboard_run_id_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GENAI_ONBOARD_RUN_ID", raising=False)
    monkeypatch.delenv("GENAI_PIPELINE_RUN_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_JOB_RUN_ID", raising=False)
    p = tmp_path / "inputs.toml"
    p.write_text(
        textwrap.dedent(
            """
            [institution]
            id = "synthetic_univ_alpha"

            [datasets.files]
            student = "a.csv"
            """
        ).strip(),
        encoding="utf-8",
    )
    raw = IdentityAgentInputsConfig.model_validate(from_toml_file(str(p)))
    school = raw.to_school_mapping_config(
        uc_catalog="dev_sst_02", onboard_run_id="run_xyz"
    )
    assert school.onboard_run_id == "run_xyz"
    assert school.pipeline_version == resolve_pipeline_version()
    root = school.genai_versioned_run_root()
    assert root is not None
    assert "run_xyz" in root
    assert root.endswith("/genai_pipeline/run_xyz")


def test_to_school_mapping_config_onboard_run_id_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GENAI_ONBOARD_RUN_ID", raising=False)
    monkeypatch.setenv("GENAI_PIPELINE_RUN_ID", "from_env")
    monkeypatch.delenv("DATABRICKS_JOB_RUN_ID", raising=False)
    p = tmp_path / "inputs.toml"
    p.write_text(
        textwrap.dedent(
            """
            [institution]
            id = "synthetic_univ_alpha"

            [datasets.files]
            student = "a.csv"
            """
        ).strip(),
        encoding="utf-8",
    )
    raw = IdentityAgentInputsConfig.model_validate(from_toml_file(str(p)))
    school = raw.to_school_mapping_config(uc_catalog="dev_sst_02")
    assert school.onboard_run_id == "from_env"
