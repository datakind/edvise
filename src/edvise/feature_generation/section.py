import logging
import typing as t

import pandas as pd

from .column_names import SectionFeatureSpec

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    *,
    section_id_cols: list[str],
    spec: SectionFeatureSpec | None = None,
) -> pd.DataFrame:
    """
    Compute section-level features from pdp course dataset w/ added course-level features,
    and add as columns to ``df`` .

    Args:
        df
        section_id_cols: Columns that uniquely identify sections, used to group course rows
            and merge section features back in.

    Note:
        Rows for which any value in ``section_id_cols`` is null won't have features
        computed. This is because such a group is "undefined" in some sense,
        so we can't know if the resulting features are accurate.

    See Also:
        - :func:`pdp.features.course.add_features()`
    """
    LOGGER.info("adding section features ...")
    s = spec or SectionFeatureSpec.all()
    if not any(
        (
            s.section_num_students_enrolled,
            s.section_num_students_passed,
            s.section_num_students_completed,
            s.section_course_grade_numeric_mean,
        )
    ):
        return df
    agg_map: dict[str, t.Any] = {}
    if s.section_num_students_enrolled:
        agg_map["section_num_students_enrolled"] = section_num_students_enrolled_col_agg()
    if s.section_num_students_passed:
        agg_map["section_num_students_passed"] = section_num_students_passed_col_agg()
    if s.section_num_students_completed:
        agg_map[
            "section_num_students_completed"
        ] = section_num_students_completed_col_agg()
    if s.section_course_grade_numeric_mean:
        agg_map["section_course_grade_numeric_mean"] = (
            section_course_grade_numeric_mean_col_agg()
        )
    df_section = (
        df.groupby(by=section_id_cols, as_index=False, observed=True, dropna=True)
        .agg(**agg_map)
    )
    if s.section_course_grade_numeric_mean and "section_course_grade_numeric_mean" in df_section.columns:
        df_section = df_section.astype(
            {"section_course_grade_numeric_mean": "Float32"}
        )
    return pd.merge(df, df_section, on=section_id_cols, how="left")


def section_num_students_enrolled_col_agg(col: str = "student_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, "count")


def section_num_students_passed_col_agg(col: str = "course_passed") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def section_num_students_completed_col_agg(
    col: str = "course_completed",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def section_course_grade_numeric_mean_col_agg(
    col: str = "course_grade_numeric",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")
