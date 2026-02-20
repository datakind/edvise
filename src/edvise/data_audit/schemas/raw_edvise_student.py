# ruff: noqa: F821
# mypy: ignore-errors
"""
Edvise cohort (student) schema for raw uploads.

Matches the edvise institution schema in edvise-api (edvise_schema_extension.json
institutions.edvise.data_models.student). Used so API and pipelines share the same
validation rules. Column names and checks align with the JSON spec and the
DataKind cohort file requirements.
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

# Student-specific patterns (edvise_schema_extension.json student model)
STUDENT_AGE_PATTERN = re.compile(
    r"(?i)^((1[3-9]|[2-9][0-9]|100)|20\s+and\s+younger|older\s+than\s+24|>20\s*-\s*24)$"
)
ENROLLMENT_TYPE_PATTERN = re.compile(
    r"(?i).*(first[-\s]?time|freshman|transfer|re[-\s]?admit|readmit).*"
)

PellYesNoField = pda.Field(nullable=True, isin=["Y", "Yes", "N", "No"])
# When present, values should be >= 0 (ge=0.0). Enforced where supported by Pandera.
CreditsEarnedField = pda.Field(nullable=True, ge=0.0)

# Max distinct values per column (cardinality limits per product spec)
MAX_CARDINALITY_GENDER = 5
MAX_CARDINALITY_CREDENTIAL_TYPE = 5
MAX_CARDINALITY_FIRST_GEN = 3


def _max_distinct_values(series: pd.Series, max_n: int) -> bool:
    """Return True if series has at most max_n distinct non-null values."""
    return series.dropna().nunique() <= max_n


class RawEdviseStudentDataSchema(pda.DataFrameModel):
    """
    Schema for raw Edvise cohort (student) data.

    Validates column presence, dtypes, and value rules per the Edvise extension
    and DataKind cohort file requirements. The DataFrame must contain all
    columns defined below; columns marked optional may contain nulls.

    Required (non-null, format-checked): student_id, enrollment_type,
    credential_type_sought_year_1, program_of_study_term_1. Optional columns
    may be null. Cardinality limits: gender and credential_type_sought_year_1
    at most 5 distinct values each; first_gen at most 3.
    """

    # Required
    student_id: pt.Series["string"] = StudentIdField
    enrollment_type: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=ENROLLMENT_TYPE_PATTERN,
    )
    credential_type_sought_year_1: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=CREDENTIAL_DEGREE_PATTERN,
    )
    program_of_study_term_1: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
    )

    # Optional
    cohort_year: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        str_matches=YEAR_PATTERN,
    )
    cohort_term: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        str_matches=TERM_PATTERN,
    )
    first_enrollment_date: pt.Series["datetime64[ns]"] = pda.Field(
        nullable=True,
    )
    student_age: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        str_matches=STUDENT_AGE_PATTERN,
    )
    race: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    ethnicity: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    gender: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    first_gen: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    pell_status_first_year: pt.Series[pd.StringDtype] = PellYesNoField
    incarcerated_status: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    military_status: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    employment_status: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    disability_status: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    first_bachelors_grad_date: pt.Series["datetime64[ns]"] = pda.Field(
        nullable=True,
    )
    first_associates_grad_date: pt.Series["datetime64[ns]"] = pda.Field(
        nullable=True,
    )
    degree_grad: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        str_matches=CREDENTIAL_DEGREE_PATTERN,
    )
    major_grad: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    certificate1_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    certificate2_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    certificate3_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    credits_earned_ap: pt.Series["float64"] = CreditsEarnedField
    credits_earned_dual_enrollment: pt.Series["float64"] = CreditsEarnedField

    @pda.check("gender", name="max_5_values")
    @classmethod
    def gender_max_5_values(cls, series: pd.Series) -> bool:
        return _max_distinct_values(series, MAX_CARDINALITY_GENDER)

    @pda.check("first_gen", name="max_3_values")
    @classmethod
    def first_gen_max_3_values(cls, series: pd.Series) -> bool:
        return _max_distinct_values(series, MAX_CARDINALITY_FIRST_GEN)

    @pda.check("credential_type_sought_year_1", name="max_5_values")
    @classmethod
    def credential_type_max_5_values(cls, series: pd.Series) -> bool:
        return _max_distinct_values(series, MAX_CARDINALITY_CREDENTIAL_TYPE)

    class Config:
        coerce = True
        strict = False
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["student_id"]
