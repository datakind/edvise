# ruff: noqa: F821
# mypy: ignore-errors
"""
Edvise course schema for raw uploads.

Matches the edvise institution schema in edvise-api (edvise_schema_extension.json
institutions.edvise.data_models.course). Used so API and pipelines share the same
validation rules. Column names and checks align with the JSON spec and the
DataKind course file requirements.
"""

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
    PELL_CATEGORIES,
    TERM_CATEGORIES,
    _apply_course_schema_transforms,
    StudentIdField,
    YEAR_PATTERN,
)

# Letter grades and non-GPA status codes per product spec
ALLOWED_LETTER_GRADES = {
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
}

CreditsField = pda.Field(nullable=False, ge=0.0)


class RawEdviseCourseDataSchema(pda.DataFrameModel):
    """
    Schema for raw Edvise course data.

    Validates column presence, dtypes, and value rules per the Edvise extension
    and DataKind course file requirements. Only required columns must be
    present; optional columns may be missing or null.

    Required (must be present, non-null, format-checked): learner_id,
    academic_year, academic_term, course_prefix, course_number, course_title,
    course_section_id, grade, course_credits_attempted, course_credits_earned.
    Optional columns may be missing from the DataFrame or contain nulls; when
    present they are validated. Rows must be unique on (learner_id,
    academic_year, academic_term, course_prefix, course_number, section_id).
    """

    # ------------------------------------------------------------------ #
    # Required
    # ------------------------------------------------------------------ #
    learner_id: pt.Series["string"] = StudentIdField
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
    course_section_id: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
    grade: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
    course_credits_attempted: pt.Series["float64"] = CreditsField
    course_credits_earned: pt.Series["float64"] = CreditsField

    # ------------------------------------------------------------------ #
    # Optional (column may be missing; when present, validated)
    # ------------------------------------------------------------------ #
    course_title: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    department: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    instructional_format: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    academic_level: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    course_begin_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    course_end_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(nullable=True)
    instructional_modality: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    gen_ed_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    prerequisite_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    instructor_appointment_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    gateway_or_developmental_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    section_size: t.Optional[pt.Series["float64"]] = pda.Field(nullable=True, ge=0.0)
    term_degree: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    term_declared_major: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    intent_to_transfer_flag: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    term_pell_recipient: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True,
        dtype_kwargs={"categories": PELL_CATEGORIES},
        coerce=True,
    )

    # ------------------------------------------------------------------ #
    # Custom checks
    # ------------------------------------------------------------------ #
    @pda.check("grade", name="valid_grade")
    @classmethod
    def grade_is_valid(cls, series: pd.Series) -> pd.Series:
        """
        Accept letter/status grades from ALLOWED_LETTER_GRADES or any numeric
        float in [0.0, 4.0] (e.g. "3.5", "2.0", "0").
        """

        def _is_valid(val: str) -> bool:
            if pd.isna(val):
                return True
            s = str(val).strip().upper()
            if s in ALLOWED_LETTER_GRADES:
                return True
            try:
                return 0.0 <= float(s) <= 4.0
            except (ValueError, TypeError):
                return False

        return series.apply(_is_valid)

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
        """Normalize academic_term and term_pell_recipient before validation."""
        check_obj = _apply_course_schema_transforms(check_obj)
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
            "learner_id",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "course_section_id",
        ]
