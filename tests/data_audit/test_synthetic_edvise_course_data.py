"""Tests for synthetic Edvise course data generator."""

import random

import pandas as pd

from edvise.data_audit.schemas import RawEdviseCourseDataSchema
from edvise.scripts.generate_synthetic_edvise_course_data import (
    build_course_rows_for_learner,
    generate_course_dataframe,
)


def test_build_course_rows_for_learner_validates() -> None:
    rng = random.Random(0)
    rows = build_course_rows_for_learner(
        learner_id="100001",
        entry_year="2022-23",
        entry_term="Fall",
        intended_program_type="Associate Degree",
        declared_major_at_entry="Biology",
        pell_recipient_year1="Y",
        rng=rng,
    )
    assert len(rows) >= 1
    schema = RawEdviseCourseDataSchema.to_schema()
    df = pd.DataFrame(rows).reindex(columns=list(schema.columns.keys()))
    RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_generate_course_dataframe_multi_students() -> None:
    df_stu = pd.DataFrame(
        [
            {
                "learner_id": "200001",
                "entry_year": "2023-24",
                "entry_term": "FALL",
                "intended_program_type": "Certificate",
                "declared_major_at_entry": "Business",
                "pell_recipient_year1": "N",
            },
            {
                "learner_id": "200002",
                "entry_year": "2023-24",
                "entry_term": "Spring",
                "intended_program_type": "Bachelor's Degree",
                "declared_major_at_entry": "Computer Science",
                "pell_recipient_year1": None,
            },
        ]
    )
    df_c = generate_course_dataframe(df_stu, seed=7, validate_schema=True)
    assert len(df_c) >= 2
    assert set(df_c["learner_id"].unique()) == {"200001", "200002"}
    assert df_c.duplicated(
        subset=[
            "learner_id",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "course_section_id",
        ]
    ).sum() == 0
