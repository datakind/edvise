"""
Column-presence helpers and cross-step feature dependencies for feature generation.

Upstream specs (:mod:`feature_resolution`) decide which features *should* be built from
raw inputs. Downstream steps (:mod:`student_term`, :mod:`cumulative`) must only consume
columns that actually exist on the frame — ES institutions often omit optional PDP fields.
"""

from __future__ import annotations

import pandas as pd

from .column_names import CourseFeatureSpec, SectionFeatureSpec


def columns_present(df: pd.DataFrame, *names: str) -> bool:
    """True when every name is a column on ``df``."""
    return bool(names) and all(name in df.columns for name in names)


def enable_when(spec_flag: bool, df: pd.DataFrame, *required_columns: str) -> bool:
    """Combine a resolved spec toggle with a runtime column-presence check."""
    return spec_flag and columns_present(df, *required_columns)


def filter_named_aggs(
    df: pd.DataFrame, aggs: dict[str, pd.NamedAgg]
) -> dict[str, pd.NamedAgg]:
    """Keep only NamedAgg entries whose source column exists on ``df``."""
    return {key: named for key, named in aggs.items() if named.column in df.columns}


# --- Downstream inputs required by student-term aggregate / add steps ---

MULTICOL_GRADE_COLUMNS = (
    "grade",
    "course_grade_numeric",
    "section_course_grade_numeric_mean",
)

SECTION_STUDENT_FRACTION_COLUMNS = (
    "sections_num_students_enrolled",
    "sections_num_students_passed",
    "sections_num_students_completed",
)

STUDENT_RATE_VS_SECTION_COLUMNS = (
    "frac_courses_passed",
    "frac_courses_completed",
    "frac_sections_students_passed",
    "frac_sections_students_completed",
)


def multicol_grade_enabled(
    *,
    course_flags: CourseFeatureSpec,
    section_flags: SectionFeatureSpec,
) -> bool:
    """Grade-vs-section aggregates need course grades and section-level grade means."""
    return bool(
        course_flags.course_grade
        and course_flags.course_grade_numeric
        and section_flags.section_course_grade_numeric_mean
    )
