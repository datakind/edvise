"""
Standardized **course** column names for :func:`~edvise.feature_generation.course.add_features`.

PDP uses Clearinghouse-style ``course_prefix`` / ``course_number``. Edvise raw course
schema uses the same identifiers (see :mod:`edvise.data_audit.schemas.raw_edvise_course`);
silver ES rows keep those names, so ES uses the same bundle as PDP here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourseStandardizedColumns:
    """Physical columns on ``df_course_standardized`` used by course feature pipes."""

    course_prefix_col: str
    course_number_col: str
    course_cip_col: str
    grade_col: str


PDP_COURSE_STANDARDIZED_COLUMNS = CourseStandardizedColumns(
    course_prefix_col="course_prefix",
    course_number_col="course_number",
    course_cip_col="course_cip",
    grade_col="grade",
)

ES_COURSE_STANDARDIZED_COLUMNS = CourseStandardizedColumns(
    course_prefix_col="course_prefix",
    course_number_col="course_number",
    course_cip_col="course_cip",
    grade_col="grade",
)
