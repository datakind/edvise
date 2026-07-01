"""Tests for GenAI batch ingest path overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from edvise.configs.genai import DatasetConfig, SchoolMappingConfig
from edvise.genai.mapping.shared.batch_input_paths import (
    apply_bronze_batch_dir_overrides,
)
from edvise.utils.gcs import SUCCESS_FILENAME


def _school_config() -> SchoolMappingConfig:
    return SchoolMappingConfig(
        institution_id="test_school",
        bronze_volumes_path="/Volumes/dev/test_school_bronze/bronze_volume",
        datasets={
            "student": DatasetConfig(files=["genai_mapping/onboard/students.csv"]),
            "course": DatasetConfig(files=["genai_mapping/onboard/courses.csv"]),
        },
    )


def test_apply_bronze_batch_dir_overrides_noop_when_empty() -> None:
    cfg = _school_config()
    assert apply_bronze_batch_dir_overrides(cfg, bronze_batch_dir="") is cfg


def test_apply_bronze_batch_dir_overrides_resolves_by_basename(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "students.csv").write_text("a\n", encoding="utf-8")
    (batch / "courses.csv").write_text("b\n", encoding="utf-8")
    (batch / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")

    updated = apply_bronze_batch_dir_overrides(
        _school_config(),
        bronze_batch_dir=str(batch),
    )
    assert updated.datasets["student"].files == [str(batch / "students.csv")]
    assert updated.datasets["course"].files == [str(batch / "courses.csv")]


def test_apply_bronze_batch_dir_overrides_resolves_by_file_kind_suffix(
    tmp_path: Path,
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "1782516108693_2026_01_20_Edvise Student File.csv").write_text(
        "a\n", encoding="utf-8"
    )
    (batch / "1782516108691_2026_01_20_Edvise Course File.csv").write_text(
        "b\n", encoding="utf-8"
    )
    (batch / SUCCESS_FILENAME).write_text("{}", encoding="utf-8")

    cfg = SchoolMappingConfig(
        institution_id="city_cols_of_chicago",
        bronze_volumes_path="/Volumes/dev/city_cols_of_chicago_bronze/bronze_volume",
        datasets={
            "student": DatasetConfig(
                files=["2025-09-19_CCC Student File.csv"],
            ),
            "course": DatasetConfig(
                files=["2025-09-19_CCC Course File.csv"],
            ),
        },
    )
    updated = apply_bronze_batch_dir_overrides(
        cfg,
        bronze_batch_dir=str(batch),
    )
    assert updated.datasets["student"].files == [
        str(batch / "1782516108693_2026_01_20_Edvise Student File.csv")
    ]
    assert updated.datasets["course"].files == [
        str(batch / "1782516108691_2026_01_20_Edvise Course File.csv")
    ]


def test_apply_bronze_batch_dir_overrides_missing_file_raises(tmp_path: Path) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    with pytest.raises(FileNotFoundError, match="student"):
        apply_bronze_batch_dir_overrides(
            _school_config(),
            bronze_batch_dir=str(batch),
        )
