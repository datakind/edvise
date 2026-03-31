"""Tests for synthetic Edvise student data generator."""

import pandas as pd

import pytest

from edvise.data_audit.schemas import RawEdviseStudentDataSchema
from edvise.scripts.generate_synthetic_edvise_student_data import (
    generate_student_dataframe,
    generate_student_row,
)


def test_generate_student_row_has_required_columns() -> None:
    row = generate_student_row(learner_id="id1", use_optionals=False)
    required = {
        "learner_id",
        "enrollment_type",
        "intended_program_type",
        "declared_major_at_entry",
    }
    for k in required:
        assert k in row
    assert row["learner_id"] is not None and row["learner_id"] != ""


def test_generate_student_row_minimal_validates() -> None:
    row = generate_student_row(learner_id="s1", use_optionals=False)
    schema = RawEdviseStudentDataSchema.to_schema()
    columns = list(schema.columns.keys())
    df = pd.DataFrame([row]).reindex(columns=columns)
    RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_generate_student_dataframe_validates() -> None:
    df = generate_student_dataframe(10, seed=42, use_optionals=True, ensure_cardinality=True)
    assert len(df) == 10
    assert df["learner_id"].is_unique
    RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_generate_student_dataframe_cardinality() -> None:
    df = generate_student_dataframe(50, seed=123, use_optionals=True, ensure_cardinality=True)
    assert df["gender"].dropna().nunique() <= 5
    assert df["first_generation_status"].dropna().nunique() <= 3
    assert df["intended_program_type"].dropna().nunique() <= 5
