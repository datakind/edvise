import logging

import pandas as pd

from edvise.feature_generation.pipeline_columns import SectionPipelineColumns

LOGGER = logging.getLogger(__name__)

_DEFAULT_SECTION_COLS = SectionPipelineColumns(
    section_id_cols=("term_id", "course_id", "section_id"),
)


def add_features(
    df: pd.DataFrame,
    *,
    columns: SectionPipelineColumns = _DEFAULT_SECTION_COLS,
) -> pd.DataFrame:
    """
    Compute section-level features from pdp course dataset w/ added course-level features,
    and add as columns to ``df`` .

    Args:
        df
        columns: Section groupby keys and source column names for aggregates.

    Note:
        Rows for which any value in ``columns.section_id_cols`` is null won't have features
        computed. This is because such a group is "undefined" in some sense,
        so we can't know if the resulting features are accurate.

    See Also:
        - :func:`pdp.features.course.add_features()`
    """
    LOGGER.info("adding section features ...")
    section_id_cols = list(columns.section_id_cols)
    df_section = (
        df.groupby(by=section_id_cols, as_index=False, observed=True, dropna=True)
        # generating named aggs via functions gives at least *some* testability
        .agg(
            section_num_students_enrolled=section_num_students_enrolled_col_agg(
                columns.student_id_col
            ),
            section_num_students_passed=section_num_students_passed_col_agg(
                columns.course_passed_col
            ),
            section_num_students_completed=section_num_students_completed_col_agg(
                columns.course_completed_col
            ),
            section_course_grade_numeric_mean=section_course_grade_numeric_mean_col_agg(
                columns.course_grade_numeric_col
            ),
        )
        .astype({"section_course_grade_numeric_mean": "Float32"})
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
