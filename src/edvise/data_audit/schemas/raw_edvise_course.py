# ruff: noqa: F821
# mypy: ignore-errors
"""
Edvise course schema for raw uploads.

Matches the edvise institution schema in edvise-api (edvise_schema_extension.json
institutions.edvise.data_models.course). Used so API and pipelines share the same
validation rules. Column names and checks align with the JSON spec and the
DataKind course file requirements.
"""

import logging
import re

import pandas as pd

try:
    import pandera as pda
    import pandera.typing as pt
except ModuleNotFoundError:
    import edvise.utils as utils

    utils.databricks.mock_pandera()
    import pandera as pda
    import pandera.typing as pt

from edvise.data_audit.schemas._edvise_shared import (
    CREDENTIAL_DEGREE_PATTERN,
    StudentIdField,
    TERM_PATTERN,
    YEAR_PATTERN,
)

LOGGER = logging.getLogger(__name__)

# Course-specific pattern (edvise_schema_extension.json course model)
TERM_ENROLLMENT_PATTERN = re.compile(r"(?i).*(full[\s-]?time|part[\s-]?time).*")

# Allowed grades per JSON spec
ALLOWED_GRADES = [
    "A+",
    "A",
    "A-",
    "B+",
    "B",
    "B-",
    "C+",
    "C",
    "C-",
    "D+",
    "D",
    "D-",
    "F",
    "P",
    "PASS",
    "S",
    "SAT",
    "U",
    "UNSAT",
    "W",
    "WD",
    "I",
    "IP",
    "AU",
    "NG",
    "NR",
    "M",
    "O",
    "0",
    "1",
    "2",
    "3",
    "4",
]
PASS_FAIL_VALUES = ["Fail", "Pass", "P", "F"]
PELL_YES_NO = ["Y", "Yes", "N", "No"]

CreditsField = pda.Field(nullable=False, ge=0.0)


class RawEdviseCourseDataSchema(pda.DataFrameModel):
    """
    Schema for raw Edvise course data.

    Validates column presence, dtypes, and value rules per the Edvise extension
    and DataKind course file requirements. The DataFrame must contain all
    columns defined below; optional columns may contain nulls.

    Required (non-null, format-checked): student_id, academic_year,
    academic_term, course_prefix, course_number, course_name, grade,
    course_credits_attempted, course_credits_earned, pass_fail_flag. Rows must
    be unique on (student_id, academic_year, academic_term, course_prefix,
    course_number).
    """

    # Required
    student_id: pt.Series["string"] = StudentIdField
    academic_year: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=YEAR_PATTERN,
    )
    academic_term: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=TERM_PATTERN,
    )
    course_prefix: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
    course_number: pt.Series["float64"] = pda.Field(nullable=False)
    course_name: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
    grade: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        isin=ALLOWED_GRADES,
    )
    course_credits_attempted: pt.Series["float64"] = CreditsField
    course_credits_earned: pt.Series["float64"] = CreditsField
    pass_fail_flag: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        isin=PASS_FAIL_VALUES,
    )

    # Optional
    department: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    course_classification: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    course_type: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    course_begin_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    course_end_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    delivery_method: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    core_course: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    prerequisite_course_flag: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
    )
    course_instructor_employment_status: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
    )
    gateway_or_development_flag: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
    )
    course_section_size: pt.Series["float64"] = pda.Field(nullable=True, ge=0.0)
    term_enrollment_intensity: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        str_matches=TERM_ENROLLMENT_PATTERN,
    )
    term_degree: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        str_matches=CREDENTIAL_DEGREE_PATTERN,
    )
    term_major: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    intent_to_transfer_flag: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
    )
    term_pell_recipient: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        isin=PELL_YES_NO,
    )

    class Config:
        coerce = True
        strict = False
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = [
            "student_id",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
        ]
