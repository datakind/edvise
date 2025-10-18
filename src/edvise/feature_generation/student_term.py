import functools as ft
import logging
import typing as t

import numpy as np
import pandas as pd


from edvise.utils.data_cleaning import convert_to_snake_case
from . import constants, term, shared

LOGGER = logging.getLogger(__name__)


def aggregate_from_course_level_features(
    df: pd.DataFrame,
    *,
    student_term_id_cols: list[str],
    min_passing_grade: float = constants.DEFAULT_MIN_PASSING_GRADE,
    key_course_subject_areas: t.Optional[list[str]] = None,
    key_course_ids: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate course-level features up to student-term-level features
    by grouping on ``student_term_id_cols`` , then aggregating columns' values
    by specified functions or as dummy columns whose values are, in turn, summed.

    Args:
        df
        student_term_id_cols: Columns that uniquely identify student-terms,
            used to group rows in ``df`` and merge features back in.
        min_passing_grade: Minimum numeric grade considered by institution as "passing".
            Default value is 1.0, i.e. a "D" grade or better.
        key_course_subject_areas: List of course subject areas that are particularly
            relevant ("key") to the institution, such that features are computed to
            measure the number of courses falling within them per student-term.
        key_course_ids

    See Also:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
        - https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#built-in-aggregation-methods

    Notes:
        Rows for which any value in ``student_term_id_cols`` is null are dropped
        and features aren't computed! This is because such a group is "undefined",
        so we can't know if the resulting features are correct.
    """
    LOGGER.info("aggregating course-level data to student-term-level features ...")
    df_grped = df.groupby(by=student_term_id_cols, observed=True, as_index=False)
    # pass through useful metadata and term features as-is
    # assumed to have the same values for every row per group
    df_passthrough = df_grped.agg(
        institution_id=("institution_id", "first"),
        academic_year=("academic_year", "first"),
        academic_term=("academic_term", "first"),
        term_start_dt=("term_start_dt", "first"),
        term_rank=("term_rank", "first"),
        term_rank_core=("term_rank_core", "first"),
        term_rank_noncore=("term_rank_noncore", "first"),
        term_is_core=("term_is_core", "first"),
        term_is_noncore=("term_is_noncore", "first"),
        term_in_peak_covid=("term_in_peak_covid", "first"),
        term_program_of_study=("term_program_of_study", "first"),
    )
    # various aggregations, with an eye toward cumulative features downstream
    df_aggs = df_grped.agg(
        num_courses=num_courses_col_agg(),
        num_courses_passed=num_courses_passed_col_agg(),
        num_courses_completed=num_courses_completed_col_agg(),
        num_credits_attempted=num_credits_attempted_col_agg(),
        num_credits_earned=num_credits_earned_col_agg(),
        course_ids=course_ids_col_agg(),
        course_subjects=course_subjects_col_agg(),
        course_subject_areas=course_subject_areas_col_agg(),
        course_id_nunique=course_id_nunique_col_agg(),
        course_subject_nunique=course_subject_nunique_col_agg(),
        course_subject_area_nunique=course_subject_area_nunique_col_agg(),
        course_level_mean=course_level_mean_col_agg(),
        course_level_std=course_level_std_col_agg(),
        course_grade_numeric_mean=course_grade_numeric_mean_col_agg(),
        course_grade_numeric_std=course_grade_numeric_std_col_agg(),
        section_num_students_enrolled_mean=section_num_students_enrolled_mean_col_agg(),
        section_num_students_enrolled_std=section_num_students_enrolled_std_col_agg(),
        sections_num_students_enrolled=sections_num_students_enrolled_col_agg(),
        sections_num_students_passed=sections_num_students_passed_col_agg(),
        sections_num_students_completed=sections_num_students_completed_col_agg(),
    )
    df_dummies = sum_dummy_cols_by_group(
        df,
        grp_cols=student_term_id_cols,
        agg_cols=[
            "course_type",
            "delivery_method",
            "math_or_english_gateway",
            "co_requisite_course",
            "course_instructor_employment_status",
            "course_instructor_rank",
            "course_level",
            "course_grade",
        ],
    )

    agg_col_vals: list[tuple[str, t.Any | list[t.Any]]] = [
        ("core_course", "Y"),
        ("course_type", ["CC", "CD"]),
        ("course_level", [0, 1]),
        ("enrolled_at_other_institution_s", "Y"),
    ]
    if key_course_subject_areas is not None:
        agg_col_vals.extend(
            ("course_subject_area", kcsa) for kcsa in key_course_subject_areas
        )
    if key_course_ids is not None:
        agg_col_vals.extend(("course_id", kc) for kc in key_course_ids)
    df_val_equals = sum_val_equal_cols_by_group(
        df, grp_cols=student_term_id_cols, agg_col_vals=agg_col_vals
    )
    df_dummy_equals = equal_cols_by_group(
        df=df_val_equals, grp_cols=student_term_id_cols
    )
    df_grade_aggs = multicol_grade_aggs_by_group(
        df, min_passing_grade=min_passing_grade, grp_cols=student_term_id_cols
    )
    return shared.merge_many_dataframes(
        [
            df_passthrough,
            df_aggs,
            df_val_equals,
            df_dummy_equals,
            df_dummies,
            df_grade_aggs,
        ],
        on=student_term_id_cols,
    )


def add_features(
    df: pd.DataFrame,
    *,
    min_num_credits_full_time: float = constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
) -> pd.DataFrame:
    """
    Compute various student-term-level features from aggregated course-level features
    joined to student-level features.

    Args:
        df
        min_num_credits_full_time: Minimum number of credits *attempted* per term
            for a student's enrollment intensity to be considered "full-time".
            Default value is 12.0.

    See Also:
        - :func:`aggregate_from_course_level_features()`
    """
    LOGGER.info("adding student-term features ...")
    nc_prefix = constants.NUM_COURSE_FEATURE_COL_PREFIX
    fc_prefix = constants.FRAC_COURSE_FEATURE_COL_PREFIX
    _num_course_cols = (
        [col for col in df.columns if col.startswith(f"{nc_prefix}_")]
        +
        # also include num-course cols to be added below
        [
            "num_courses_in_program_of_study_area_term_1",
            "num_courses_in_program_of_study_area_year_1",
            "num_courses_in_term_program_of_study_area",
        ]
    )
    num_frac_courses_cols = [
        (col, col.replace(f"{nc_prefix}_", f"{fc_prefix}_")) for col in _num_course_cols
    ]
    feature_name_funcs = (
        {
            "year_of_enrollment_at_cohort_inst": year_of_enrollment_at_cohort_inst,
            "student_has_earned_certificate_at_cohort_inst": ft.partial(
                student_earned_certificate, inst="cohort"
            ),
            "student_has_earned_certificate_at_other_inst": ft.partial(
                student_earned_certificate, inst="other"
            ),
            "term_is_pre_cohort": term_is_pre_cohort,
            "term_is_while_student_enrolled_at_other_inst": term_is_while_student_enrolled_at_other_inst,
            "term_program_of_study_area": term_program_of_study_area,
            "frac_credits_earned": shared.frac_credits_earned,
            "student_term_enrollment_intensity": ft.partial(
                student_term_enrollment_intensity,
                min_num_credits_full_time=min_num_credits_full_time,
            ),
            "num_courses_in_program_of_study_area_term_1": ft.partial(
                num_courses_in_study_area,
                study_area_col="student_program_of_study_area_term_1",
            ),
            "num_courses_in_program_of_study_area_year_1": ft.partial(
                num_courses_in_study_area,
                study_area_col="student_program_of_study_area_year_1",
            ),
            "num_courses_in_term_program_of_study_area": ft.partial(
                num_courses_in_study_area,
                study_area_col="term_program_of_study_area",
            ),
        }
        | {
            fc_col: ft.partial(compute_frac_courses, numer_col=nc_col)
            for nc_col, fc_col in num_frac_courses_cols
        }
        | {
            "frac_sections_students_passed": ft.partial(
                compute_frac_sections_students,
                numer_col="sections_num_students_passed",
            ),
            "frac_sections_students_completed": ft.partial(
                compute_frac_sections_students,
                numer_col="sections_num_students_completed",
            ),
        }
        | {
            "student_pass_rate_above_sections_avg": ft.partial(
                student_rate_above_sections_avg,
                student_col="frac_courses_passed",
                sections_col="frac_sections_students_passed",
            ),
            "student_completion_rate_above_sections_avg": ft.partial(
                student_rate_above_sections_avg,
                student_col="frac_courses_completed",
                sections_col="frac_sections_students_completed",
            ),
        }
    )
    return df.assign(**feature_name_funcs)


def year_of_enrollment_at_cohort_inst(
    df: pd.DataFrame,
    *,
    cohort_start_dt_col: str = "cohort_start_dt",
    term_start_dt_col: str = "term_start_dt",
) -> pd.Series:
    dts_diff = (df[term_start_dt_col].sub(df[cohort_start_dt_col])).dt.days
    return pd.Series(np.ceil((dts_diff + 1) / 365.25), dtype="Int8")


def student_earned_certificate(
    df: pd.DataFrame,
    *,
    inst: t.Literal["cohort", "other"],
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    degree_year_cols = [
        f"first_year_to_certificate_at_{inst}_inst",
        f"years_to_latest_certificate_at_{inst}_inst",
    ]
    return df.loc[:, degree_year_cols].lt(df[enrollment_year_col], axis=0).any(axis=1)


def term_is_pre_cohort(
    df: pd.DataFrame,
    *,
    cohort_start_dt_col: str = "cohort_start_dt",
    term_start_dt_col: str = "term_start_dt",
) -> pd.Series:
    return df[term_start_dt_col].lt(df[cohort_start_dt_col]).astype("boolean")


# TODO: we could probably compute this directly, w/o an intermediate feature?
def term_is_while_student_enrolled_at_other_inst(
    df: pd.DataFrame, *, col: str = "num_courses_enrolled_at_other_institution_s_Y"
) -> pd.Series:
    return df[col].gt(0)


def term_program_of_study_area(
    df: pd.DataFrame, *, col: str = "term_program_of_study"
) -> pd.Series:
    return shared.extract_short_cip_code(df[col])


def num_courses_in_study_area(
    df: pd.DataFrame,
    *,
    study_area_col: str,
    course_subject_areas_col: str = "course_subject_areas",
    fill_value: str = "-1",
) -> pd.Series:
    return (
        pd.DataFrame(df[course_subject_areas_col].tolist(), dtype="string")
        .eq(df[study_area_col].fillna(fill_value), axis="index")
        .sum(axis="columns")
        .astype("Int8")
    )


def compute_frac_courses(
    df: pd.DataFrame, *, numer_col: str, denom_col: str = "num_courses"
) -> pd.Series:
    result = df[numer_col].div(df[denom_col])
    if not result.between(0.0, 1.0, inclusive="both").all():
        raise ValueError()
    return result


def compute_frac_sections_students(
    df: pd.DataFrame,
    *,
    numer_col: str,
    denom_col: str = "sections_num_students_enrolled",
) -> pd.Series:
    result = df[numer_col].div(df[denom_col])
    if not result.between(0.0, 1.0, inclusive="both").all():
        raise ValueError()
    return result


def student_rate_above_sections_avg(
    df: pd.DataFrame, *, student_col: str, sections_col: str
) -> pd.Series:
    return df[student_col].gt(df[sections_col])


def student_term_enrollment_intensity(
    df: pd.DataFrame,
    *,
    min_num_credits_full_time: float,
    num_credits_col: str = "num_credits_attempted",
) -> pd.Series:
    if df[num_credits_col].isna().any():
        LOGGER.warning(
            "%s null values found for '%s'; "
            "calculation of student_term_enrollment_intensity doesn't correctly handle nulls",
            df[num_credits_col].isna().sum(),
            num_credits_col,
        )
    return pd.Series(
        data=np.where(
            df[num_credits_col].ge(min_num_credits_full_time), "FULL-TIME", "PART-TIME"
        ),
        index=df.index,
        dtype="string",
    )


def num_courses_col_agg(col: str = "course_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, "count")


def num_courses_passed_col_agg(col: str = "course_passed") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def num_courses_completed_col_agg(col: str = "course_completed") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def num_credits_attempted_col_agg(
    col: str = "number_of_credits_attempted",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def num_credits_earned_col_agg(col: str = "number_of_credits_earned") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def course_ids_col_agg(col: str = "course_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, _agg_values_in_list)


def course_subjects_col_agg(col: str = "course_cip") -> pd.NamedAgg:
    return pd.NamedAgg(col, _agg_values_in_list)


def course_subject_areas_col_agg(col: str = "course_subject_area") -> pd.NamedAgg:
    return pd.NamedAgg(col, _agg_values_in_list)


def _agg_values_in_list(ser: pd.Series) -> list:
    result = ser.tolist()
    assert isinstance(result, list)  # type guard
    return result


def course_id_nunique_col_agg(col: str = "course_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, "nunique")


def course_subject_nunique_col_agg(col: str = "course_cip") -> pd.NamedAgg:
    return pd.NamedAgg(col, "nunique")


def course_subject_area_nunique_col_agg(
    col: str = "course_subject_area",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "nunique")


def course_level_mean_col_agg(col: str = "course_level") -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")


def course_level_std_col_agg(col: str = "course_level") -> pd.NamedAgg:
    return pd.NamedAgg(col, "std")


def course_grade_numeric_mean_col_agg(col: str = "course_grade_numeric") -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")


def course_grade_numeric_std_col_agg(col: str = "course_grade_numeric") -> pd.NamedAgg:
    return pd.NamedAgg(col, "std")


def section_num_students_enrolled_mean_col_agg(
    col: str = "section_num_students_enrolled",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")


def section_num_students_enrolled_std_col_agg(
    col: str = "section_num_students_enrolled",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "std")


def sections_num_students_enrolled_col_agg(
    col: str = "section_num_students_enrolled",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def sections_num_students_passed_col_agg(
    col: str = "section_num_students_passed",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def sections_num_students_completed_col_agg(
    col: str = "section_num_students_completed",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def sum_dummy_cols_by_group(
    df: pd.DataFrame, *, grp_cols: list[str], agg_cols: list[str]
) -> pd.DataFrame:
    """
    Compute dummy values for all ``agg_cols`` in ``df`` , then group by ``grp_cols``
    and aggregate by "sum" to get the number of values for each dummy value.

    Args:
        df
        grp_cols
        agg_cols
    """
    return (
        pd.get_dummies(
            df[grp_cols + agg_cols],
            columns=agg_cols,
            sparse=False,
            dummy_na=False,
            drop_first=False,
        )
        .groupby(by=grp_cols, observed=True, as_index=True)
        .agg("sum")
        .rename(columns=_rename_sum_by_group_col)
        .reset_index(drop=False)
    )


def equal_cols_by_group(
    df: pd.DataFrame,
    *,
    grp_cols: list[str],
) -> pd.DataFrame:
    """
    Compute dummy values for all of the num_course features

    Args:
        df
        grp_cols
    """
    num_prefix = constants.NUM_COURSE_FEATURE_COL_PREFIX
    dummy_prefix = constants.DUMMY_COURSE_FEATURE_COL_PREFIX

    course_subject_prefixes = [
        constants.NUM_COURSE_FEATURE_COL_PREFIX + "_course_id",
        constants.NUM_COURSE_FEATURE_COL_PREFIX + "_course_subject_area",
    ]

    dummy_cols = {
        col.replace(num_prefix, dummy_prefix, 1): df[col].ge(1)
        for col in df.columns
        if any(col.startswith(prefix) for prefix in course_subject_prefixes)
    }

    return df.assign(**dummy_cols).reindex(columns=grp_cols + list(dummy_cols.keys()))


def sum_val_equal_cols_by_group(
    df: pd.DataFrame,
    *,
    grp_cols: list[str],
    agg_col_vals: list[tuple[str, t.Any]],
) -> pd.DataFrame:
    """
    Compute equal to specified values for all ``agg_col_vals`` in ``df`` ,
    then group by ``grp_cols`` and aggregate with a "sum".

    Args:
        df
        grp_cols
        agg_col_vals
    """
    temp_col_series = {}
    for col, val in agg_col_vals:
        # make multi-value col names nicer to read
        temp_col = (
            f"{col}_{'|'.join(str(item) for item in val)}"
            if isinstance(val, list)
            else f"{col}_{val}"
        )
        temp_col_series[temp_col] = shared.compute_values_equal(df[col], val)
    return (
        df.assign(**temp_col_series)
        .reindex(columns=grp_cols + list(temp_col_series.keys()))
        .groupby(by=grp_cols, observed=True, as_index=True)
        .agg("sum")
        .rename(columns=_rename_sum_by_group_col)
        .reset_index(drop=False)
    )


def _rename_sum_by_group_col(col: str) -> str:
    return f"{constants.NUM_COURSE_FEATURE_COL_PREFIX}_{col}"


def multicol_grade_aggs_by_group(
    df: pd.DataFrame,
    *,
    min_passing_grade: float,
    grp_cols: list[str],
    grade_col: str = "grade",
    grade_numeric_col: str = "course_grade_numeric",
    section_grade_numeric_col: str = "section_course_grade_numeric_mean",
) -> pd.DataFrame:
    return (
        df.loc[:, grp_cols + [grade_col, grade_numeric_col, section_grade_numeric_col]]
        # compute intermediate column values all at once, which is efficient
        .assign(
            course_grade_is_failing_or_withdrawal=ft.partial(
                _course_grade_is_failing_or_withdrawal,
                min_passing_grade=min_passing_grade,
                grade_col=grade_col,
                grade_numeric_col=grade_numeric_col,
            ),
            course_grade_above_section_avg=ft.partial(
                _course_grade_above_section_avg,
                grade_numeric_col=grade_numeric_col,
                section_grade_numeric_col=section_grade_numeric_col,
            ),
        )
        .groupby(by=grp_cols, observed=True, as_index=False)
        # so that we can efficiently aggregate those intermediate values per group
        .agg(
            num_courses_grade_is_failing_or_withdrawal=(
                "course_grade_is_failing_or_withdrawal",
                "sum",
            ),
            num_courses_grade_above_section_avg=(
                "course_grade_above_section_avg",
                "sum",
            ),
        )
    )


def add_historical_features_student_term_data(
    df, student_id_col, sort_cols, gpa_cols, num_cols
):
    """
    *CUSTOM SCHOOL FUNCTION*

    Append cumulative and rolling historical features to student-term level data

    Args:
        df (pd.DataFrame): data unique at the student-term level
        student_id_col (str): name of column containing student IDs
        sort_cols (list[str]): time or date-based columns to sort by. If the dataset contains
            a date for the observation, enter this column only in the list.
            If the dataset does not contain a date, include the year column as the first element of
            the list, and the season column as the second element of the list. Year should be in
            the format YYYY-YY of the academic year, and season should be one of Fall, Winter, Spring,
            Summer
        gpa_cols (list[str]): names of columns containing GPA data
        num_cols (list[str]): names of other numeric columns. Note that columns starting with
            'nunique_' and {constants.TERM_COURSE_SUM_PREFIX} are already included in the aggregations.

    Note that any modification to these list arguments within the function will change
    the objects permanently, if used outside of the function.

    Returns:
        pd.DataFrame: student-term level data, with appended cumulative and rolling historical features
    """

    if len(sort_cols) == 2:
        year_col = sort_cols[0]
        season_col = sort_cols[1]
        df["term_end_date"] = df.apply(
            lambda x: term.create_term_end_date(x[year_col], x[season_col]), axis=1
        )
        sort_cols += ["term_end_date"]

    sorted_df = df.sort_values([student_id_col] + sort_cols, ascending=True)

    # term number - note that this is the student's term number and they do not necessarily indicate consecutive terms enrolled
    sorted_df[constants.TERM_NUMBER_COL] = (
        sorted_df.groupby([student_id_col]).cumcount() + 1
    )

    num_cols.append(constants.TERM_N_COURSES_ENROLLED_COLNAME)
    num_cols = list(set(num_cols))

    # Note: The old way of implementing has been buggy:
    # groupby().expanding().agg({column1: [fn1, fn2, fn3],
    #                            column2: [fn1, fn3, fn4]})
    # Since DataFrame.groupby().transform() does not accept a dictionary like DataFrame.transform() does,
    # implementing this more verbose way for accuracy, even though there may be a more concise way of doing this
    cumul_min_df = sorted_df.groupby(student_id_col)[num_cols + gpa_cols].transform(
        "cummin"
    )  # lowest X so far
    cumul_min_df.columns = [
        constants.MIN_PREFIX + orig_col + constants.HIST_SUFFIX
        for orig_col in cumul_min_df.columns.values
    ]

    cumul_max_df = sorted_df.groupby(student_id_col)[num_cols + gpa_cols].transform(
        "cummax"
    )  # highest X so far
    cumul_max_df.columns = [
        constants.MAX_PREFIX + orig_col + constants.HIST_SUFFIX
        for orig_col in cumul_max_df.columns.values
    ]

    # adding this only for numeric columns separate from GPA columns because it doesn't make sense to get a cumulative sum of GPA
    dummy_sum_cols = [
        dummy_sum_col
        for dummy_sum_col in df.columns.values
        if (
            dummy_sum_col.startswith(
                (constants.TERM_COURSE_SUM_PREFIX, constants.TERM_FLAG_PREFIX)
            )
            and (dummy_sum_col not in num_cols + gpa_cols)
        )
    ]
    cumul_sum_df = sorted_df.groupby(student_id_col)[
        list(set(num_cols + dummy_sum_cols))
    ].transform("cumsum")  # total X so far
    cumul_sum_df.columns = [
        (
            orig_col.replace(
                constants.TERM_COURSE_SUM_PREFIX, constants.TERM_COURSE_SUM_HIST_PREFIX
            )
            + constants.HIST_SUFFIX
            if orig_col.startswith(constants.TERM_COURSE_SUM_PREFIX)
            else (
                orig_col.replace(
                    constants.TERM_FLAG_PREFIX, constants.TERM_FLAG_SUM_HIST_PREFIX
                )
                + constants.HIST_SUFFIX
                if orig_col.startswith(constants.TERM_FLAG_PREFIX)
                else constants.SUM_PREFIX + orig_col + constants.HIST_SUFFIX
            )
        )
        for orig_col in cumul_sum_df.columns.values
    ]

    mean_cols = [
        col
        for col in sorted_df.columns.values
        if col.startswith(("term_nunique_", f"term_{constants.MEAN_NAME}"))
    ]
    cumul_avg_df = (
        sorted_df.groupby(student_id_col)[num_cols + gpa_cols + mean_cols]
        .expanding()
        .mean()
        .reset_index()
    )
    cumul_avg_df = cumul_avg_df.drop(columns=[student_id_col, "level_1"])
    cumul_avg_df.columns = [
        constants.MEAN_NAME + "_" + orig_col + constants.HIST_SUFFIX
        for orig_col in cumul_avg_df.columns.values
    ]

    # TODO: rolling std for num_cols + gpa_cols
    # def std(x):  # ensure that we are using population standard deviation
    #     return np.std(x, ddof=0)

    student_term_hist_df = pd.concat(
        [sorted_df, cumul_min_df, cumul_max_df, cumul_sum_df, cumul_avg_df], axis=1
    )

    # make sure no rows got dropped
    assert (
        cumul_max_df.shape[0]
        == cumul_min_df.shape[0]
        == cumul_sum_df.shape[0]
        == cumul_avg_df.shape[0]
        == sorted_df.shape[0]
        == student_term_hist_df.shape[0]
    )

    # Identify new feature columns
    new_feature_cols = set(student_term_hist_df.columns) - set(sorted_df.columns)

    # Convert only new feature columns those to snake_case
    student_term_hist_df.rename(
        columns={col: convert_to_snake_case(col) for col in new_feature_cols},
        inplace=True,
    )

    return student_term_hist_df


def course_data_to_student_term_level(
    df, groupby_cols, count_unique_cols, sum_cols, mean_cols, dummy_cols
):
    """
    *CUSTOM SCHOOL FUNCTION*

    Convert student-course data to student-term data by aggregating the following:

    Args:
        df (pd.DataFrame): data unique by student and course
        groupby_cols (list[str]): names of groupby columns for aggregation i.e. term columns,
            student IDs and characteristics, any semester-level variables that
            do not change across courses within a semester
        count_unique_cols (list[str]): names of columns to count(distinct) values from at the
            student-term level. Consider categorical columns with many possible values.
        sum_cols (list[str]): names of numeric columns to sum to the student-term level,
            for example - course credit columns (attempted, earned, etc.)
        mean_cols (list[str]): names of numeric columns to calculate the mean at the student-term level,
            for example - total class size
        dummy_cols (list[str]): names of categorical columns to convert into dummy variables and
            then sum up to the student-term level. Consider categorical columns with few possible
            levels (<5-10) or whose values (although many) may have a significant impact on the outcome
            variable.

    Note that any modification to these list arguments within the function will change
    the objects permanently, if used outside of the function.

    This was developed with NSC data in mind. From our call with NSC,
    we learned that data at the course level is less common, so we made
    intentional choices to make less detailed calculations at the course level,
    because we are thinking about how to generalize this module to other schools
    and data outside of NSC. We could consider adding more defined course-level
    features and aggregations later. For example, the "largest" course in terms
    of credits attempted over time, but right now this adds a lot of complexity
    to engineering historical features in a later processing step.

    Returns:
        pd.DataFrame: data unique by student and term
    """

    n_courses = (
        df.groupby(groupby_cols)
        .size()
        .reset_index()
        .rename(columns={0: constants.TERM_N_COURSES_ENROLLED_COLNAME})
    )

    dummies_df = (
        pd.get_dummies(df[groupby_cols + dummy_cols], dummy_na=True, columns=dummy_cols)
        .groupby(groupby_cols)
        .agg(["sum", "mean"])
    )
    dummies_df.columns = [
        (
            (constants.TERM_COURSE_SUM_PREFIX + category)
            if fn == "sum"
            else (constants.TERM_COURSE_PROP_PREFIX + category)
        )
        for category, fn in dummies_df.columns
    ]

    derived_cols_dict = {
        col: [] for col in set(count_unique_cols + sum_cols + mean_cols)
    }
    for count_unique_col in count_unique_cols:
        derived_cols_dict[count_unique_col].append(("nunique", pd.Series.nunique))
    for sum_col in sum_cols:
        derived_cols_dict[sum_col].append((constants.SUM_NAME, "sum"))
    for mean_col in mean_cols:
        derived_cols_dict[mean_col].append((constants.MEAN_NAME, "mean"))
    derived_df = df.groupby(groupby_cols).agg(derived_cols_dict).reset_index()
    if derived_df.columns.nlevels == 2:
        derived_df.columns = [
            f"term_{col2}_{col1}" if col2 != "" else col1
            for col1, col2 in derived_df.columns
        ]

    student_term_df = n_courses.merge(dummies_df.reset_index(), on=groupby_cols).merge(
        derived_df, on=groupby_cols
    )

    # Identify new feature columns
    new_feature_cols = set(student_term_df.columns) - set(df.columns)

    # Convert only new feature columns those to snake_case
    student_term_df.rename(
        columns={col: convert_to_snake_case(col) for col in new_feature_cols},
        inplace=True,
    )

    return student_term_df


def calculate_pct_terms_unenrolled(
    student_term_df,
    possible_terms_list,
    new_col_prefix,
    term_rank_col="term_rank",
    student_id_col="student_id",
):
    """
    *CUSTOM SCHOOL FUNCTION*

    Calculate percent of a student's terms unenrolled to date.

    Args:
        student_term_df (pd.DataFrame): data at the student + term level, containing columns term_rank_col and student_id_col
        possible_terms_list (list): list of possible terms to consider. This can be all possible terms or a subset of terms,
            for example - only Fall/Spring terms
        new_col_prefix (str): prefix of new column indicating percent of a student's terms unenrolled to date, according to the scope of possible_terms_list
        term_rank_col (str, optional): column name of the column containing the student's term rank. Defaults to 'term_rank'.
        student_id_col (str, optional): column name of the column containing student IDs. Defaults to 'Student.ID'.

    Returns:
        pd.DataFrame: student_term_df with a new column, new_col_prefix + hist_suffix
    """
    student_term_df["term_ranks_enrolled_to_date"] = (
        student_term_df.sort_values(term_rank_col)
        .groupby(student_id_col)[term_rank_col]
        .transform(_cumulative_list_aggregation)
    )
    student_term_df["first_enrolled_term_rank"] = [
        min(enrolled_ranks)
        for enrolled_ranks in student_term_df["term_ranks_enrolled_to_date"]
    ]
    student_term_df["possible_term_ranks_to_date"] = [
        [
            rank
            for rank in possible_terms_list
            if (row["first_enrolled_term_rank"] <= rank <= row[term_rank_col])
        ]
        for _, row in student_term_df.iterrows()
    ]
    student_term_df["skipped_term_ranks_to_date"] = [
        [
            rank
            for rank in row["possible_term_ranks_to_date"]
            if (rank not in row["term_ranks_enrolled_to_date"])
        ]
        for _, row in student_term_df.iterrows()
    ]

    possible_n_terms_col = (
        constants.TERM_FLAG_SUM_HIST_PREFIX + "possible" + constants.HIST_SUFFIX
    )
    unenrolled_n_terms_col = (
        constants.TERM_FLAG_SUM_HIST_PREFIX + "unenrolled" + constants.HIST_SUFFIX
    )
    student_term_df[possible_n_terms_col] = [
        len(possible_term_ranks)
        for possible_term_ranks in student_term_df["possible_term_ranks_to_date"]
    ]
    student_term_df[unenrolled_n_terms_col] = [
        len(skipped_term_ranks)
        for skipped_term_ranks in student_term_df["skipped_term_ranks_to_date"]
    ]
    student_term_df[new_col_prefix + constants.HIST_SUFFIX] = (
        student_term_df[unenrolled_n_terms_col] / student_term_df[possible_n_terms_col]
    )
    student_term_df = student_term_df.drop(
        columns=[
            "term_ranks_enrolled_to_date",
            "first_enrolled_term_rank",
            "possible_term_ranks_to_date",
            "skipped_term_ranks_to_date",
            possible_n_terms_col,
            unenrolled_n_terms_col,
        ]
    )
    return student_term_df


def calculate_avg_credits_rolling(df, date_col, student_id_col, credit_col, n_days):
    """
    *CUSTOM SCHOOL FUNCTION*

    Calculate average credits per term enrolled within a time period.

    Args:
        df (pd.DataFrame): term dataset containing date_col, student_id_col, and credit_col
        date_col (str): column name of column containing dates of terms to use in windowing function
        student_id_col (str): column name of column containing student ID, for grouping the windowing function
        credit_col (str): column name of column containing term-level credits to aggregate
        n_days (int): number of days to calculate average credits across

    Returns:
        pd.DataFrame: df with new columns rolling_credit_col, rolling_n_terms_col, and rolling_avg_col
    """
    window = f"{n_days}d"
    rolling_credit_col = (
        constants.SUM_PREFIX + credit_col + "_" + window + constants.HIST_SUFFIX
    )
    rolling_n_terms_col = "n_terms_enrolled_" + window + constants.HIST_SUFFIX
    rolling_avg_col = (
        f"{constants.MEAN_NAME}_{credit_col}_" + window + constants.HIST_SUFFIX
    )

    # When rolling() is specified by an integer window, min_periods defaults to this integer.
    # That means if, for example, we wanted to calculate something across a window of 4 terms,
    # the rolling window is null early in the student's history when they don't yet have 4 terms.
    # We use a timedelta window instead to avoid this.
    grouped_rolling_df = (
        df.sort_values(date_col)
        .groupby(student_id_col)
        .rolling(window=window, on=date_col)
    )

    attempted_credits_last_365 = (
        grouped_rolling_df[credit_col]
        .sum()
        .reset_index()
        .rename(columns={credit_col: rolling_credit_col})
    )
    terms_enrolled_last_365 = grouped_rolling_df[date_col].count()
    terms_enrolled_last_365.name = rolling_n_terms_col

    merged_df = (
        df.merge(attempted_credits_last_365, on=[student_id_col, date_col])
        .merge(terms_enrolled_last_365.reset_index(), on=[student_id_col, date_col])
        .sort_values([student_id_col, date_col])
    )
    merged_df[rolling_avg_col] = (
        merged_df[rolling_credit_col] / merged_df[rolling_n_terms_col]
    )

    return merged_df


def _course_grade_is_failing_or_withdrawal(
    df: pd.DataFrame,
    min_passing_grade: float,
    grade_col: str = "grade",
    grade_numeric_col: str = "course_grade_numeric",
) -> pd.Series:
    return (
        df[grade_col].isin({"F", "W"})
        | df[grade_numeric_col].between(0.0, min_passing_grade, inclusive="left")
    )  # fmt: skip


def _course_grade_above_section_avg(
    df: pd.DataFrame,
    grade_numeric_col: str = "course_grade_numeric",
    section_grade_numeric_col: str = "section_course_grade_numeric_mean",
) -> pd.Series:
    return df[grade_numeric_col].gt(df[section_grade_numeric_col])


def _cumulative_list_aggregation(list_values_over_time):
    """
    *CUSTOM SCHOOL FUNCTION*

    Aggregate column values over time into
    cumulative lists. Example use case: we have a student term dataset
    containing the term numbers a student is enrolled in. Applying this
    function to that dataset using
    df.sort_values(term_rank_col).groupby(student_id_col).transform(cumulative_list_aggregation)
    would give us, at each term, a list of the terms a student has been enrolled in
    to date.

    Args:
        list_values_over_time (list): list of values to accumulate over time

    Returns:
        list of lists
    """
    out = [[]]
    for x in list_values_over_time:
        out.append(out[-1] + [x])
    return out[1:]
