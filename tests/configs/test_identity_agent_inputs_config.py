"""Tests for :class:`edvise.configs.genai.IdentityAgentInputsConfig`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from edvise.configs.genai import DatasetConfig, IdentityAgentInputsConfig
from edvise.dataio.read import from_toml_file


def test_identity_agent_inputs_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "inputs.toml"
    p.write_text(
        textwrap.dedent(
            """
            [institution]
            id = "john_jay_col"

            [files]
            student = "Datakind Students.1994-2025.csv"
            course = [
              "Datakind Student Classes.2005-2013.csv",
              "Datakind Student Classes.2014-2025.csv",
            ]
            semester = "Datakind Student Terms-2015 Fall-2025 Summer II.csv"
            """
        ).strip(),
        encoding="utf-8",
    )

    raw = IdentityAgentInputsConfig.model_validate(from_toml_file(str(p)))
    assert raw.institution.id == "john_jay_col"
    assert raw.files["student"] == "Datakind Students.1994-2025.csv"
    assert raw.files["course"] == [
        "Datakind Student Classes.2005-2013.csv",
        "Datakind Student Classes.2014-2025.csv",
    ]

    school = raw.to_school_mapping_config()
    assert school.institution_id == "john_jay_col"
    assert school.datasets["student"] == DatasetConfig(
        files=["Datakind Students.1994-2025.csv"],
        primary_keys=None,
    )
    assert school.datasets["course"].files == [
        "Datakind Student Classes.2005-2013.csv",
        "Datakind Student Classes.2014-2025.csv",
    ]
    assert school.datasets["course"].primary_keys is None


def test_files_rejects_non_string_list() -> None:
    with pytest.raises(ValidationError):
        IdentityAgentInputsConfig.model_validate(
            {
                "institution": {"id": "x"},
                "files": {"a": [1, 2]},
            }
        )


def test_dataset_config_rejects_empty_primary_keys_when_set() -> None:
    with pytest.raises(ValueError, match="primary_keys"):
        DatasetConfig(files=["/a.csv"], primary_keys=[])
