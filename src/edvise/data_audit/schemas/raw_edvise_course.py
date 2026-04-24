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

# Manifest + SMA execution: these target keys must map (ENTITY_GRAIN in
# manifest.validation). ``course_section_id`` / ``source_term_key`` are optional
# when an institution omits them; composite row uniqueness for pipeline output is
# enforced separately (see ``course_output_uniqueness_key_columns``) because Pandera
# ``Config.unique`` cannot express "include column in the key only if present".
COURSE_MANIFEST_GRAIN_KEYS: tuple[str, ...] = (
    "learner_id",
    "academic_year",
    "academic_term",
    "course_prefix",
    "course_number",
)

# Optional disambiguators merged when mapped (not in COURSE_MANIFEST_GRAIN_KEYS).
# See field_executor._derive_entity_keys.
COURSE_OPTIONAL_GRAIN_TARGETS: tuple[str, ...] = ("source_term_key",)


def course_output_uniqueness_key_columns(df: pd.DataFrame) -> list[str]:
    """
    Target-space columns that jointly identify a course output row for uniqueness.

    Always includes :data:`COURSE_MANIFEST_GRAIN_KEYS`. Appends ``source_term_key``
    and ``course_section_id`` only when those columns exist on ``df`` (SMA may omit
    unmapped targets entirely, so missing column means course-level / no IA key).
    """
    keys = list(COURSE_MANIFEST_GRAIN_KEYS)
    if "source_term_key" in df.columns:
        keys.append("source_term_key")
    if "course_section_id" in df.columns:
        keys.append("course_section_id")
    return keys


def course_output_row_uniqueness_violation_message(
    df: pd.DataFrame,
    *,
    max_sample_rows: int = 5,
) -> str | None:
    """
    If ``df`` has duplicate rows under :func:`course_output_uniqueness_key_columns`,
    return a short diagnostic string; otherwise ``None``.

    Intended for SMA / pipeline logging (avoids Pandera lazy multi-million failure_cases
    when only uniqueness is wrong).
    """
    if df.empty:
        return None
    keys = [k for k in course_output_uniqueness_key_columns(df) if k in df.columns]
    missing_base = [k for k in COURSE_MANIFEST_GRAIN_KEYS if k not in df.columns]
    if missing_base:
        return (
            "Cannot check course row uniqueness: missing base columns "
            f"{missing_base!r} (have {list(df.columns)!r})"
        )
    dup = df.duplicated(subset=keys, keep=False)
    if not dup.any():
        return None
    n = int(dup.sum())
    sample = df.loc[dup, keys].head(max_sample_rows)
    return (
        f"{n} row(s) duplicate on uniqueness keys {keys!r} "
        f"(sample up to {max_sample_rows} rows):\n{sample.to_string()}"
    )


class RawEdviseCourseDataSchema(pda.DataFrameModel):
    """
    Schema for raw Edvise course data.

    Validates column presence, dtypes, and value rules per the Edvise extension
    and DataKind course file requirements. Only required columns must be
    present; optional columns may be missing or null.

    Required (must be present, non-null, format-checked): learner_id,
    academic_year, academic_term, course_prefix, course_number,
    grade, course_credits_attempted, course_credits_earned.
    Optional columns (e.g. course_title, course_section_id, source_term_key) may be
    missing from the DataFrame or contain nulls; when present they are validated.
    Pandera does **not** enforce composite row uniqueness on this model
    (``Config.unique`` is empty): Pandera cannot express optional key columns such as
    ``course_section_id`` / ``source_term_key`` that participate in the grain only when
    present. Call :func:`course_output_row_uniqueness_violation_message` on pipeline
    outputs after :meth:`validate`. When ``source_term_key`` is supplied (e.g. after IdentityAgent term
    normalization), SMA may include it in the execution grain via
    :data:`COURSE_OPTIONAL_GRAIN_TARGETS` (see field_executor._derive_entity_keys).
    """

    # ------------------------------------------------------------------ #
    # Required
    # ------------------------------------------------------------------ #
    learner_id: pt.Series[pd.StringDtype] = StudentIdField
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
    source_term_key: pt.Series[pd.StringDtype] = pda.Field(
        nullable=False,
        str_length={"min_value": 1},
        description=(
            "Stable key for the source term instance (e.g. concat of raw year, season, "
            "and term order). Used in the uniqueness grain so enrollments stay distinct "
            "when academic_year/academic_term are canonicalized."
        ),
    )
    course_section_id: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        description="Catalog section when available; optional when not provided by the institution.",
    )
    grade: pt.Series[pd.StringDtype] = pda.Field(nullable=False)
    course_credits_attempted: pt.Series[pd.Float64Dtype] = CreditsField
    course_credits_earned: pt.Series[pd.Float64Dtype] = CreditsField

    # ------------------------------------------------------------------ #
    # Optional (column may be missing; when present, validated)
    # ------------------------------------------------------------------ #
    source_term_key: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True,
        str_length={"min_value": 1},
        description=(
            "Stable key for the source term instance (e.g. _term_grain from term "
            "normalization). Optional for direct Edvise uploads; when present and mapped "
            "in SMA, included in execution entity grain (see COURSE_OPTIONAL_GRAIN_TARGETS)."
        ),
    )
    course_title: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    department: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    instructional_format: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(
        nullable=True
    )
    academic_level: t.Optional[pt.Series[pd.StringDtype]] = pda.Field(nullable=True)
    course_begin_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(nullable=True)
    course_end_date: t.Optional[pt.Series[pt.DateTime]] = pda.Field(nullable=True)
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
    course_section_size: t.Optional[pt.Series[pd.Float64Dtype]] = pda.Field(
        nullable=True, ge=0.0
    )
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
        # Composite uniqueness: use course_output_row_uniqueness_violation_message —
        # Pandera unique= cannot omit optional key columns when unmapped.
        unique: list[str] = []
