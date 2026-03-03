"""Tests for synthetic Edvise student data generator."""

import pandas as pd

import pytest

from edvise.data_audit.schemas import RawEdviseStudentDataSchema
from edvise.scripts.generate_synthetic_edvise_student_data import (
    generate_student_dataframe,
    generate_student_row,
)


def test_generate_student_row_has_required_columns() -> None:
    row = generate_student_row(student_id="id1", use_optionals=False)
    required = {"student_id", "enrollment_type", "credential_type_sought_year_1", "program_of_study_term_1"}
    for k in required:
        assert k in row
        assert row[k] is not None and row[k] != ""


def test_generate_student_row_minimal_validates() -> None:
    row = generate_student_row(student_id="s1", use_optionals=False)
    schema = RawEdviseStudentDataSchema.to_schema()
    columns = list(schema.columns.keys())
    df = pd.DataFrame([row]).reindex(columns=columns)
    RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_generate_student_dataframe_validates() -> None:
    df = generate_student_dataframe(10, seed=42, use_optionals=True, ensure_cardinality=True)
    assert len(df) == 10
    assert df["student_id"].is_unique
    RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_generate_student_dataframe_cardinality() -> None:
    df = generate_student_dataframe(50, seed=123, use_optionals=True, ensure_cardinality=True)
    assert df["gender"].dropna().nunique() <= 5
    assert df["first_gen"].dropna().nunique() <= 3
    assert df["credential_type_sought_year_1"].nunique() <= 5
