# ruff: noqa: F821
# mypy: ignore-errors
"""
Edvise course schema for raw uploads.

Matches the edvise institution schema in edvise-api (edvise_schema_extension.json
institutions.edvise.data_models.course). Used so API and pipelines share the same
validation rules. Column names and checks align with the JSON spec and the
DataKind course file requirements.
"""

import re
import typing as t

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
    TERM_CATEGORIES,
    _apply_edvise_pdp_transforms_course,
    StudentIdField,
    YEAR_PATTERN,
)

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
    and DataKind course file requirements. Only required columns must be
    present; optional columns may be missing or null.

    Required (must be present, non-null, format-checked): student_id,
    academic_year, academic_term, course_prefix, course_number, course_name,
    grade, course_credits_attempted, course_credits_earned, pass_fail_flag.
    Optional columns may be missing from the DataFrame or contain nulls; when
    present they are validated. Rows must be unique on (student_id,
    academic_year, academic_term, course_prefix, course_number).
    """

    # Required
    student_id: pt.Series["string"] = StudentIdField
    academic_year: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=YEAR_PATTERN,
    )
    academic_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=False,
        dtype_kwargs={"categories": TERM_CATEGORIES, "ordered": True},
        coerce=True,
    )
    course_prefix: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
    course_number: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
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

    # Optional (column may be missing; when present, validated)
    department: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    course_classification: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    course_type: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    course_begin_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    course_end_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(nullable=True)
    delivery_method: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    core_course: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    prerequisite_course_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    course_instructor_employment_status: t.Optional[pt.Series[pd.StringDtype]] = (
        pda.Field(nullable=True)
    )
    gateway_or_development_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    course_section_size: t.Optional[pt.Series["float64"]] = pda.Field(
        nullable=True, ge=0.0
    )
    term_enrollment_intensity: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        str_matches=TERM_ENROLLMENT_PATTERN,
    )
    term_major: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    intent_to_transfer_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    term_pell_recipient: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        isin=PELL_YES_NO,
    )

    # PDP-compat: originals in raw_*; main column holds extracted/PDP value
    term_degree: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        isin=["Bachelor's", "Associate's", "Certificate"],
    )
    raw_term_degree: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    raw_grade: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)

    @classmethod
    def validate(
        cls,
        check_obj: pd.DataFrame,
        head: t.Optional[int] = None,
        tail: t.Optional[int] = None,
        sample: t.Optional[int] = None,
        random_state: t.Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """Run PDP-compat transforms then validate (so coercion sees FALL, etc.)."""
        check_obj = _apply_edvise_pdp_transforms_course(check_obj)
        return super().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
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
