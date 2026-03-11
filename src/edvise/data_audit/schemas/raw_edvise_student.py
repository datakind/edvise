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
    ENROLLMENT_CATEGORIES,
    PELL_CATEGORIES,
    TERM_CATEGORIES,
    _apply_edvise_pdp_transforms_student,
    StudentIdField,
    YEAR_PATTERN,
)

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
    and DataKind cohort file requirements. Only required columns must be
    present; optional columns may be missing or null.

    Required (must be present, non-null, format-checked): student_id,
    enrollment_type, credential_type_sought_year_1, program_of_study_term_1,
    cohort, cohort_term.
    Optional columns may be missing from the DataFrame or contain nulls; when
    present they are validated. Cardinality limits: gender and
    credential_type_sought_year_1 at most 5 distinct values each; first_gen
    at most 3.
    """

    # Required (after pre-validate transform: PDP categories; schema coerces to categorical)
    student_id: pt.Series["string"] = StudentIdField
    enrollment_type: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=False,
        dtype_kwargs={"categories": ENROLLMENT_CATEGORIES},
        coerce=True,
    )
    # After transform: PDP-style canonical (Bachelor's/Associate's/Certificate); original in raw_*
    credential_type_sought_year_1: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        isin=["Bachelor's", "Associate's", "Certificate"],
    )
    program_of_study_term_1: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
    )
    cohort: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=YEAR_PATTERN,
    )
    cohort_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=False,
        dtype_kwargs={"categories": TERM_CATEGORIES, "ordered": True},
        coerce=True,
    )

    # Optional (column may be missing; when present, validated)
    first_enrollment_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True,
    )
    # PDP-style age buckets (after transform); original preserved in raw_student_age
    student_age: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        isin=["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"],
    )
    race: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    ethnicity: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    gender: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    first_gen: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    pell_status_first_year: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
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
    first_bachelors_grad_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    first_associates_grad_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    major_grad: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    certificate1_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    certificate2_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    certificate3_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    credits_earned_ap: t.Optional[pt.Series["float64"]] = CreditsEarnedField
    credits_earned_dual_enrollment: t.Optional[pt.Series["float64"]] = (
        CreditsEarnedField
    )

    # PDP-compat: originals preserved in raw_*; main column holds extracted/PDP value
    raw_enrollment_type: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    raw_student_age: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    raw_credential_type_sought_year_1: t.Optional[pt.Series[pd.StringDtype]] = (
        pda.Field(nullable=True)
    )
    degree_grad: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        isin=["Bachelor's", "Associate's", "Certificate"],
    )
    raw_degree_grad: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)

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
        """Run PDP-compat transforms then validate (so coercion sees FALL, FIRST-TIME, etc.)."""
        check_obj = _apply_edvise_pdp_transforms_student(check_obj)
        return super().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

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


class RawEdviseStudentDataSchemaFlexible(pda.DataFrameModel):
    """
    Flexible schema for raw Edvise cohort (student) data.

    Designed for GenAI mapping workflows where exact canonical values may not be
    achievable. This schema:
    1. Uses StringDtype instead of CategoricalDtype (no fixed category constraints)
    2. Removes `isin` value constraints (allows any string values)
    3. Maintains structure, nullability, and basic type validation

    Use this schema for Agent 2 transformation map validation when strict canonical
    values are not required. The original RawEdviseStudentDataSchema remains available
    for strict validation after normalization.

    Required (must be present, non-null): student_id, enrollment_type,
    credential_type_sought_year_1, program_of_study_term_1, cohort, cohort_term.
    Optional columns may be missing from the DataFrame or contain nulls.
    """

    # Required - all changed to StringDtype, no category/isin constraints
    student_id: pt.Series["string"] = StudentIdField
    enrollment_type: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        # No categories - accepts any string value
    )
    credential_type_sought_year_1: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True,
        # No isin constraint - accepts any string value
    )
    program_of_study_term_1: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
    )
    cohort: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_matches=YEAR_PATTERN,  # Keep pattern validation for year format
    )
    cohort_term: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        # No categories - accepts any string value
    )

    # Optional - all changed to StringDtype, no isin constraints
    first_enrollment_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True,
    )
    student_age: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        # No isin constraint - accepts any string value
    )
    race: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    ethnicity: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    gender: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    first_gen: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    pell_status_first_year: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        # No categories - accepts any string value
    )
    incarcerated_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    military_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    employment_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    disability_status: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    first_bachelors_grad_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    first_associates_grad_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    major_grad: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    certificate1_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    certificate2_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    certificate3_date: t.Optional[pt.Series["datetime64[ns]"]] = pda.Field(
        nullable=True
    )
    credits_earned_ap: t.Optional[pt.Series["float64"]] = CreditsEarnedField
    credits_earned_dual_enrollment: t.Optional[pt.Series["float64"]] = (
        CreditsEarnedField
    )

    # Raw columns preserved
    raw_enrollment_type: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    raw_student_age: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    raw_credential_type_sought_year_1: t.Optional[pt.Series[pd.StringDtype]] = (
        pda.Field(nullable=True)
    )
    degree_grad: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        # No isin constraint - accepts any string value
    )
    raw_degree_grad: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)

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
        """
        Validate without PDP transforms - accepts flexible values as-is.

        Note: This skips the PDP normalization transforms since we're accepting
        flexible values. If canonical values are needed later, apply normalization
        utilities separately.
        """
        # Skip PDP transforms - validate flexible values directly
        return super().validate(
            check_obj, head, tail, sample, random_state, lazy, inplace
        )

    # Keep cardinality checks for data quality
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
