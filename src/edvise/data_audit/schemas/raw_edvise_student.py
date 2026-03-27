# ruff: noqa: F821
# mypy: ignore-errors
"""
Edvise cohort (student) schema for raw uploads.

Matches the edvise institution schema in edvise-api (edvise_schema_extension.json
institutions.edvise.data_models.student). Used so API and pipelines share the same
validation rules. Column names and checks align with the JSON spec and the
DataKind cohort file requirements.
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
    PELL_CATEGORIES,
    TERM_CATEGORIES,
    _apply_student_schema_transforms,
    StudentIdField,
    YEAR_PATTERN,
)

# Student-specific patterns (edvise_schema_extension.json student model)
STUDENT_AGE_PATTERN = re.compile(
    r"(?i)^((1[3-9]|[2-9][0-9]|100)|20\s+and\s+younger|older\s+than\s+24|>20\s*-\s*24)$"
)

PellYesNoField = pda.Field(nullable=True, isin=["Y", "Yes", "N", "No"])
# When present, values should be >= 0 (ge=0.0). Enforced where supported by Pandera.
CreditsEarnedField = pda.Field(nullable=True, ge=0.0)


class RawEdviseStudentDataSchema(pda.DataFrameModel):
    """
    Schema for raw Edvise cohort (student) data.

    Validates column presence, dtypes, and value rules per the Edvise extension
    and DataKind cohort file requirements. Only required columns must be
    present; optional columns may be missing or null.

    Required (must be present, non-null, format-checked): learner_id,
    enrollment_type, intended_program_type, declared_major_at_entry,
    entry_year, entry_term.
    Optional columns may be missing from the DataFrame or contain nulls; when
    present they are validated.
    """

    # ------------------------------------------------------------------ #
    # Required
    # ------------------------------------------------------------------ #
    learner_id: pt.Series[pd.StringDtype] = StudentIdField
    entry_year: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=YEAR_PATTERN,
    )
    entry_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=False,
        dtype_kwargs={"categories": TERM_CATEGORIES, "ordered": True},
        coerce=True,
    )
    enrollment_type: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    intended_program_type: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    declared_major_at_entry: pt.Series[pd.StringDtype] = pda.Field(nullable=True)

    # ------------------------------------------------------------------ #
    # Optional (column may be missing; when present, validated)
    # ------------------------------------------------------------------ #
    matriculation_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(
        nullable=True,
    )
    learner_age: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        isin=["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"],
    )
    race: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    ethnicity: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    gender: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    first_generation_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    pell_recipient_year1: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True,
        dtype_kwargs={"categories": PELL_CATEGORIES},
        coerce=True,
    )
    incarcerated_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    military_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    employment_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    disability_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    bachelors_degree_conferral_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(
        nullable=True
    )
    associates_degree_conferral_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(
        nullable=True
    )
    conferred_credential_type: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    major_at_completion: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    certificate1_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(nullable=True)
    certificate2_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(nullable=True)
    certificate3_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(nullable=True)
    credits_earned_ap: t.Optional[pt.Series[pd.Float64Dtype]] = CreditsEarnedField
    credits_earned_dual_enrollment: t.Optional[pt.Series[pd.Float64Dtype]] = (
        CreditsEarnedField
    )

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
        """Normalize entry_term, learner_age, and pell_recipient_year1 before validation."""
        check_obj = _apply_student_schema_transforms(check_obj)
        return super().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

    class Config:
        coerce = True
        strict = False
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["learner_id"]
