import pandas as pd
import pytest

from edvise.feature_generation import student_term

# @pytest.mark.parametrize(
#     ["df", "exp"],
#     [
#         (
#             pd.DataFrame(
#                 {
#                     "cohort": ["20-21", "20-21", "20-21", "22-23", "22-23"],
#                     "academic_year": ["20-21", "21-22", "23-24", "22-23", "23-24"],
#                     "num_credits_earned": [5.0, 10.0, 6.0, 0.0, 15.0],
#                     "num_credits_attempted": [10.0, 10.0, 8.0, 15.0, 15.0],
#                 }
#             ),
#             pd.DataFrame(
#                 {
#                     "year_of_enrollment_at_cohort_inst": [1, 2, 4, 1, 2],
#                     "frac_credits_earned": [0.5, 1.0, 0.75, 0.0, 1.0],
#                 }
#             ),
#         ),
#     ],
# )
# def test_add_student_term_features(df, exp):
#     obs = student_term.add_features(df)
#     assert isinstance(obs, pd.DataFrame) and not obs.empty
#     assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grp_cols", "agg_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "456", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "course_type": ["CU", "CD", "CU", "CU", "CC", "CU"],
                    "course_level": [1, 0, 1, 2, 0, 1],
                }
            ),
            ["student_guid", "term_id"],
            ["course_type", "course_level"],
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "num_courses_course_type_CC": [0, 0, 1, 0],
                    "num_courses_course_type_CD": [1, 0, 0, 0],
                    "num_courses_course_type_CU": [1, 1, 1, 1],
                    "num_courses_course_level_0": [1, 0, 1, 0],
                    "num_courses_course_level_1": [1, 1, 0, 1],
                    "num_courses_course_level_2": [0, 0, 1, 0],
                }
            ),
        ),
    ],
)
def test_sum_dummy_cols_by_group(df, grp_cols, agg_cols, exp):
    obs = student_term.sum_dummy_cols_by_group(df, grp_cols=grp_cols, agg_cols=agg_cols)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grp_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "num_courses_course_type_CC|CD": [0, 3, 1, 0],
                    "num_courses_course_id_eng_101": [0, 0, 0, 1],
                    "num_courses_course_subject_area_51": [2, 1, 1, 1],
                }
            ),
            ["student_guid", "term_id"],
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "took_course_id_eng_101": [False, False, False, True],
                    "took_course_subject_area_51": [True, True, True, True],
                }
            ),
        ),
    ],
)
def test_equal_cols_by_group(df, grp_cols, exp):
    obs = student_term.equal_cols_by_group(df, grp_cols=grp_cols)
    print("obs columns", obs.columns)
    print("exp cols", exp.columns)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grp_cols", "agg_col_vals", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "456", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "course_type": ["CU", "CD", "CU", "CU", "CC", "CU"],
                    "course_level": [1, 0, 1, 2, 0, 1],
                    "grade": ["F", "F", "P", "W", "F", "P"],
                }
            ),
            ["student_guid", "term_id"],
            [
                ("course_type", ["CC", "CD"]),
                ("course_level", 0),
                ("course_level", [2, 3]),
                ("grade", ["0", "1", "F", "W"]),
            ],
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "num_courses_course_type_CC|CD": [1, 0, 1, 0],
                    "num_courses_course_level_0": [1, 0, 1, 0],
                    "num_courses_course_level_2|3": [0, 0, 1, 0],
                    "num_courses_grade_0|1|F|W": [2, 0, 2, 0],
                }
            ),
        ),
    ],
)
def test_sum_val_equal_cols_by_group(df, grp_cols, agg_col_vals, exp):
    obs = student_term.sum_val_equal_cols_by_group(
        df, grp_cols=grp_cols, agg_col_vals=agg_col_vals
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    [
        "df",
        "min_passing_grade",
        "grp_cols",
        "grade_col",
        "grade_numeric_col",
        "section_grade_numeric_col",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "sid": ["123", "123", "123", "123", "456", "456"],
                    "tid": [
                        "22-23 FA",
                        "22-23 FA",
                        "22-23 FA",
                        "22-23 SP",
                        "22-23 SP",
                        "22-23 SP",
                    ],
                    "grade": ["4", "3", "F", "1", pd.NA, "4"],
                    "grade_num": [4.0, 3.0, pd.NA, 1.0, pd.NA, 4.0],
                    "section_grade_num_mean": [3.25, 3.0, 2.75, 2.5, 3.0, 3.5],
                }
            ).astype({"grade": "string", "grade_num": "Float32"}),
            1.0,
            ["sid", "tid"],
            "grade",
            "grade_num",
            "section_grade_num_mean",
            pd.DataFrame(
                {
                    "sid": ["123", "123", "456"],
                    "tid": ["22-23 FA", "22-23 SP", "22-23 SP"],
                    "num_courses_grade_is_failing_or_withdrawal": [1, 0, 0],
                    "num_courses_grade_above_section_avg": [1, 0, 1],
                }
            ),
        ),
    ],
)
def test_multicol_grade_aggs_by_group(
    df,
    min_passing_grade,
    grp_cols,
    grade_col,
    grade_numeric_col,
    section_grade_numeric_col,
    exp,
):
    obs = student_term.multicol_grade_aggs_by_group(
        df,
        min_passing_grade=min_passing_grade,
        grp_cols=grp_cols,
        grade_col=grade_col,
        grade_numeric_col=grade_numeric_col,
        section_grade_numeric_col=section_grade_numeric_col,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "ccol", "tcol", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "cohort_start_dt": ["2019-09-01", "2019-09-01", "2021-02-01"],
                    "term_start_dt": ["2020-02-01", "2020-09-01", "2023-09-01"],
                },
                dtype="datetime64[s]",
            ),
            "cohort_start_dt",
            "term_start_dt",
            pd.Series([1, 2, 3], dtype="Int8"),
        ),
    ],
)
def test_year_of_enrollment_at_cohort_inst(df, ccol, tcol, exp):
    obs = student_term.year_of_enrollment_at_cohort_inst(
        df, cohort_start_dt_col=ccol, term_start_dt_col=tcol
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "inst", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year_of_enrollment_at_cohort_inst": [1, 2, 3, 4],
                    "first_year_to_certificate_at_cohort_inst": [
                        2,
                        pd.NA,
                        2,
                        2,
                    ],
                    "years_to_latest_certificate_at_cohort_inst": [
                        3,
                        3,
                        pd.NA,
                        3,
                    ],
                    "first_year_to_certificate_at_other_inst": [
                        2,
                        pd.NA,
                        2,
                        2,
                    ],
                    "years_to_latest_certificate_at_other_inst": [
                        3,
                        3,
                        pd.NA,
                        3,
                    ],
                },
                dtype="Int8",
            ),
            "cohort",
            pd.Series([False, False, True, True], dtype="boolean"),
        ),
    ],
)
def test_student_earned_certificate(df, inst, exp):
    obs = student_term.student_earned_certificate(df, inst=inst)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert pd.testing.assert_series_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["df", "ccol", "tcol", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "cohort_start_dt": ["2019-09-01", "2019-09-01", "2021-02-01"],
                    "term_start_dt": ["2020-02-01", "2019-09-01", "2020-09-01"],
                },
                dtype="datetime64[s]",
            ),
            "cohort_start_dt",
            "term_start_dt",
            pd.Series([False, False, True], dtype="boolean"),
        ),
    ],
)
def test_term_is_pre_cohort(df, ccol, tcol, exp):
    obs = student_term.term_is_pre_cohort(
        df, cohort_start_dt_col=ccol, term_start_dt_col=tcol
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "study_area_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "study_area_term_1": ["01", "02", None, "03"],
                    "study_area_year_1": ["01", "03", None, "03"],
                    "course_subject_areas": [
                        ["01", "01", "01", "02"],
                        ["01", "02", "01"],
                        ["01", "02", "03"],
                        [],
                    ],
                }
            ).astype({"study_area_term_1": "string", "study_area_year_1": "string"}),
            "study_area_term_1",
            pd.Series([3, 1, 0, 0], dtype="Int8"),
        ),
        (
            pd.DataFrame(
                {
                    "study_area_term_1": ["01", "02", None, "03"],
                    "study_area_year_1": ["01", "03", None, "03"],
                    "course_subject_areas": [
                        ["01", "01", "01", "02"],
                        ["01", "02", "01"],
                        ["01", "02", "03"],
                        [],
                    ],
                }
            ).astype({"study_area_term_1": "string", "study_area_year_1": "string"}),
            "study_area_year_1",
            pd.Series([3, 0, 0, 0], dtype="Int8"),
        ),
    ],
)
def test_num_courses_in_study_area(df, study_area_col, exp):
    obs = student_term.num_courses_in_study_area(df, study_area_col=study_area_col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "numer_col", "denom_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "num_courses": [1, 2, 4, 5, 7],
                    "num_courses_passed": [1, 1, 3, 4, 0],
                }
            ),
            "num_courses_passed",
            "num_courses",
            pd.Series([1.0, 0.5, 0.75, 0.8, 0.0]),
        ),
    ],
)
def test_compute_frac_courses(df, numer_col, denom_col, exp):
    obs = student_term.compute_frac_courses(
        df, numer_col=numer_col, denom_col=denom_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "student_col", "sections_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "frac_courses_passed": [1.0, 0.5, 0.75],
                    "frac_sections_students_passed": [0.9, 0.5, 0.8],
                }
            ),
            "frac_courses_passed",
            "frac_sections_students_passed",
            pd.Series([True, False, False]),
        ),
    ],
)
def test_student_rate_above_sections_avg(df, student_col, sections_col, exp):
    obs = student_term.student_rate_above_sections_avg(
        df, student_col=student_col, sections_col=sections_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "min_num_credits_full_time", "num_credits_col", "exp"],
    [
        (
            pd.DataFrame({"num_credits_attempted": [15.0, 12.0, 8.0, 0.0]}),
            12.0,
            "num_credits_attempted",
            pd.Series(["FULL-TIME", "FULL-TIME", "PART-TIME", "PART-TIME"]),
        ),
    ],
)
def test_student_term_enrollment_intensity(
    df, min_num_credits_full_time, num_credits_col, exp
):
    obs = student_term.student_term_enrollment_intensity(
        df,
        min_num_credits_full_time=min_num_credits_full_time,
        num_credits_col=num_credits_col,
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


# CUSTOM SCHOOL TESTS/FIXTURES


@pytest.fixture
def nsc_student_term_data():
    return pd.DataFrame(
        {
            "student_id": [123, 123, 456],
            "academic_year": ["2010-11", "2011-12", "2010-11"],
            "term": ["Fall", "Fall", "Spring"],
            "semester_gpa": [3.5, 3.0, 2.9],
            "term_n_courses_enrolled": [2, 1, 1],
            "term_n_courses_course_prefix_eng": [1, 1, 1],
            "term_n_courses_course_prefix_mth": [1, 0, 0],
            "term_n_courses_course_prefix_nan": [0, 0, 0],
            "term_prop_courses_course_prefix_eng": [0.5, 1.0, 1.0],
            "term_prop_courses_course_prefix_mth": [0.5, 0.0, 0.0],
            "term_prop_courses_course_prefix_nan": [0.0, 0.0, 0.0],
            "term_nunique_course_prefix": [2, 1, 1],
            "term_nunique_institution_id": [1, 1, 1],
            "term_total_number_of_credits_earned": [8, 5, 2],
            "term_avg_number_of_credits_earned": [4.0, 5.0, 2.0],
        }
    )


@pytest.fixture
def nsc_student_term_hist_data(nsc_student_term_data):
    df = nsc_student_term_data.copy(deep=True)

    df["term_end_date"] = pd.to_datetime(["12-01-2010", "12-01-2011", "06-01-2011"])
    df["term_n"] = [1, 2, 1]  # TODO: add example where terms not consecutive
    df["term_flag_full_time"] = [1, 1, 1]

    # num_cols
    df["min_semester_gpa_to_date"] = [3.5, 3.0, 2.9]
    df["max_semester_gpa_to_date"] = [3.5, 3.5, 2.9]
    df["avg_semester_gpa_to_date"] = [3.5, 3.25, 2.9]
    ## TODO: add back in once we add back to the function
    # df["cumul_std_Semester/Session GPA_to_date"] = [0, 0.25, 0]

    df["min_term_n_courses_enrolled_to_date"] = [2, 1, 1]
    df["max_term_n_courses_enrolled_to_date"] = [2, 2, 1]
    # df["cumul_std_n_courses_enrolled"] = [0, 0.5, 0]
    df["total_n_courses_enrolled_to_date"] = [2, 3, 1]
    df["avg_term_n_courses_enrolled_to_date"] = [2.0, 1.5, 1.0]

    df["min_term_total_number_of_credits_earned_to_date"] = [8, 5, 2]
    df["max_term_total_number_of_credits_earned_to_date"] = [8, 8, 2]
    df["avg_term_total_number_of_credits_earned_to_date"] = [8.0, 6.5, 2.0]
    # df["cumul_std_Number of Credits Earned_to_date"] = [0, 1.5, 0]
    df["total_term_total_number_of_credits_earned_to_date"] = [8, 13, 2]

    # mean cols
    df["avg_term_nunique_institution_id_to_date"] = [1.0, 1.0, 1.0]
    df["avg_term_nunique_course_prefix_to_date"] = [2.0, 1.5, 1.0]
    df["avg_term_avg_number_of_credits_earned_to_date"] = [4.0, 4.5, 2.0]

    # dummy cols
    df["total_n_courses_course_prefix_eng_to_date"] = [1, 2, 1]
    df["total_n_courses_course_prefix_mth_to_date"] = [1, 1, 0]
    df["total_n_courses_course_prefix_nan_to_date"] = [0, 0, 0]
    df["n_terms_full_time_to_date"] = [1, 2, 1]

    return df


# TODO: create these dataframes in a more readable way
# fmt: off
@pytest.fixture
def nsc_course_data_sample():
    return pd.DataFrame(
        {
            "student_id": [123, 123, 123, 456],
            "academic_year": ["2010-11", "2010-11", "2011-12", "2010-11"],
            "term": ["Fall", "Fall", "Fall", "Spring"],
            "semester_gpa": [3.5, 3.5, 3.0, 2.9],
            "total_combined_earned_and_transferred_credits": [8, 8, 13, 2],
            "institution_id": [12345678, 12345678, 87654321, 12345678],
            "course_prefix": ["ENG", "MTH", "ENG", "ENG"],
            "number_of_credits_earned": [3, 5, 5, 2],
        }
    )




def test_add_historical_features_student_term_data(
    nsc_student_term_data, nsc_student_term_hist_data
):
    nsc_student_term_data["term_flag_full_time"] = [1, 1, 1]

    student_term_data_hist = student_term.add_historical_features_student_term_data(
        df=nsc_student_term_data,
        student_id_col="student_id",
        sort_cols=["academic_year", "term"],
        gpa_cols=["semester_gpa"],
        num_cols=["term_n_courses_enrolled", "term_total_number_of_credits_earned"],
    )
    pd.testing.assert_frame_equal(
        student_term_data_hist, nsc_student_term_hist_data, check_like=True
    )


def test_calculate_avg_credits_rolling(nsc_student_term_hist_data):
    orig_cols = nsc_student_term_hist_data.columns
    credit_col = "term_total_number_of_credits_earned"
    result_df = student_term.calculate_avg_credits_rolling(
        nsc_student_term_hist_data,
        date_col="term_end_date",
        student_id_col="student_id",
        credit_col=credit_col,
        n_days=400,
    )
    pd.testing.assert_frame_equal(nsc_student_term_hist_data, result_df[orig_cols])
    assert list(result_df[f"total_{credit_col}_400d_to_date"]) == [8.0, 13.0, 2.0]
    assert list(result_df["n_terms_enrolled_400d_to_date"]) == [1, 2, 1]
    assert list(result_df[f"avg_{credit_col}_400d_to_date"]) == [8.0, 6.5, 2.0]


def test_calculate_pct_terms_unenrolled():
    new_col_prefix = "pct_terms_unenrolled"
    student_term_df = pd.DataFrame(
        {
            "student": ["A", "A", "A", "B", "B"],
            "term_rank": [1, 5, 20, 1, 4],
            new_col_prefix + "_to_date": [0, 3 / 5, 17 / 20, 0, 0.5],
        }
    )
    possible_terms = list(range(1, 21))
    result_df = student_term.calculate_pct_terms_unenrolled(
        student_term_df[["student", "term_rank"]],
        possible_terms_list=possible_terms,
        new_col_prefix=new_col_prefix,
        student_id_col="student",
    )
    pd.testing.assert_frame_equal(student_term_df, result_df, check_like=False)


def test_course_data_to_student_term_level(
    nsc_course_data_sample, nsc_student_term_data
):
    groupby_cols = ["student_id", "academic_year", "term"]
    semester_cols = ["semester_gpa"]
    count_unique_cols = ["institution_id", "course_prefix"]
    num_cols = ["number_of_credits_earned"]
    cat_cols = ["course_prefix"]

    student_term_df = student_term.course_data_to_student_term_level(
        nsc_course_data_sample,
        groupby_cols=groupby_cols + semester_cols,
        count_unique_cols=count_unique_cols,
        sum_cols=num_cols,
        mean_cols=num_cols,
        dummy_cols=cat_cols,
    )
    pd.testing.assert_frame_equal(
        nsc_student_term_data, student_term_df, check_like=True
    )


def test_cumulative_list_aggregation():
    student_term_ranks_enrolled = [1, 2, 4, 5, 10]
    result = student_term._cumulative_list_aggregation(student_term_ranks_enrolled)
    assert result == [[1], [1, 2], [1, 2, 4], [1, 2, 4, 5], [1, 2, 4, 5, 10]]
