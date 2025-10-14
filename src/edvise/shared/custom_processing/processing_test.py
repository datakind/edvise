import numpy as np
import pandas as pd
import pytest

from .processing import (
    add_cumulative_nunique_col,
    add_historical_features_student_term_data,
    calculate_avg_credits_rolling,
    calculate_pct_terms_or_courses_hist,
    calculate_pct_terms_unenrolled,
    convert_number_of_courses_cols_to_term_flag_cols,
    course_data_to_student_term_level,
    create_term_end_date,
    create_terms_lkp,
    cumulative_list_aggregation,
    extract_course_level_from_course_number,
    extract_year_season,
)


# TODO
def test_clean_column_name():
    pass


def test_cumulative_list_aggregation():
    student_term_ranks_enrolled = [1, 2, 4, 5, 10]
    result = cumulative_list_aggregation(student_term_ranks_enrolled)
    assert result == [[1], [1, 2], [1, 2, 4], [1, 2, 4, 5], [1, 2, 4, 5, 10]]


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
    result_df = calculate_pct_terms_unenrolled(
        student_term_df[["student", "term_rank"]],
        possible_terms_list=possible_terms,
        new_col_prefix=new_col_prefix,
        student_id_col="student",
    )
    pd.testing.assert_frame_equal(student_term_df, result_df, check_like=False)


@pytest.mark.parametrize(
    "input,output",
    [
        ("  1234A ", 1),
        ("1234A", 1),
        (" 302H", 0),
        ("4567    ", 4),
        ("4567", 4),
        ("123", 0),
    ],
    ids=[
        "whitespace 5 digit",
        "5 digit",
        "leading whitespace 4 digit",
        "whitespace 4 digit",
        "4 digit",
        "3 digit",
    ],
)
def test_extract_course_level_from_course_number(input, output):
    assert extract_course_level_from_course_number(input) == output


@pytest.mark.parametrize(
    "invalid_course_number",
    ["   12345 ", "12345", "123456"],
    ids=[
        "whitespace 5 digit not ending in character",
        "5 digit not ending in character",
        ">5 digits",
    ],
)
def test_extract_course_level_from_course_number_raises_exception(
    invalid_course_number,
):
    with pytest.raises(Exception):
        extract_course_level_from_course_number(invalid_course_number)


def test_course_data_to_student_term_level(
    nsc_course_data_sample, nsc_student_term_data
):

    groupby_cols = ["Student ID", "Academic Year", "Term"]
    semester_cols = ["Semester/Session GPA"]
    count_unique_cols = ["Institution ID", "Course Prefix"]
    num_cols = ["Number of Credits Earned"]
    cat_cols = ["Course Prefix"]

    student_term_df = course_data_to_student_term_level(
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


def test_convert_number_of_courses_to_term_flag_col():
    test_df = pd.DataFrame(
        {
            "term_n_courses_something_A": [1, 0, 5],
            "term_n_courses_something_B": [9, 1, 0],
            "ignore_col": [1, 2, 3],
            "taking_course_something_A_this_term": [1, 0, 1],
            "taking_course_something_B_this_term": [1, 1, 0],
        }
    )
    result_df = convert_number_of_courses_cols_to_term_flag_cols(
        test_df.iloc[:, :3], col_prefix="term_n_courses_", orig_col="something"
    )
    pd.testing.assert_frame_equal(result_df, test_df.iloc[:, 2:])


def test_extract_year_season():
    term_data = pd.DataFrame(
        {
            "term": ["2022SP", "2019S1", np.nan],
            0: [2022, 2019, np.nan],
            1: ["SP", "S1", np.nan],
        }
    )
    result = extract_year_season(term_data["term"])
    assert result.tail(1).isna().all().all()
    assert result[0].head(2).tolist() == term_data[0].head(2).tolist()
    assert result[1].head(2).tolist() == term_data[1].head(2).tolist()


@pytest.mark.parametrize(
    "invalid_term",
    ["20223SP", "202FA", "2010FAA"],
    ids=["5 digit year", "3 digit year", "3 character season"],
)
def test_extract_year_season_raises_exception(invalid_term):
    invalid_df = pd.DataFrame({"term": [invalid_term]})
    with pytest.raises(Exception):
        extract_year_season(invalid_df["term"])


@pytest.mark.parametrize(
    "year,season,expected_date",
    [
        ("2011-12", "Fall", "12-01-2011"),
        ("2011-12", "Spring", "06-01-2012"),
        ("2011-12", "Summer", "08-01-2012"),
        ("2011-12", "Winter", "02-01-2012"),
    ],
)
def test_create_term_end_date(year, season, expected_date):
    assert create_term_end_date(year, season) == pd.to_datetime(expected_date)


def test_create_term_end_date_raises_exception():
    with pytest.raises(Exception):
        create_term_end_date("2011-12", "Invalid Season")


def test_create_terms_lkp():
    max_year = 2015
    min_year = 2010
    possible_seasons = pd.DataFrame(
        {"season": ["Fall", "Winter", "Spring", "S1", "S2"], "order": [1, 2, 3, 4, 5]}
    )
    terms_lkp = create_terms_lkp(min_year, max_year, possible_seasons)
    assert (
        terms_lkp.shape[0]
        == (max_year - min_year + 1) * possible_seasons.shape[0]
        == terms_lkp.term_rank.max()
    )
    assert list(terms_lkp.columns) == [
        "academic_year",
        "season",
        "order",
        "term_order",
        "calendar_year",
        "term",
        "term_end_date",
        "term_rank",
    ]
    assert terms_lkp.term_rank.min() == 1


def test_add_historical_features_student_term_data(
    nsc_student_term_data, nsc_student_term_hist_data
):
    nsc_student_term_data["term_flag_Full-Time"] = [1, 1, 1]

    student_term_data_hist = add_historical_features_student_term_data(
        df=nsc_student_term_data,
        student_id_col="Student ID",
        sort_cols=["Academic Year", "Term"],
        gpa_cols=["Semester/Session GPA"],
        num_cols=["term_n_courses_enrolled", "term_total_Number of Credits Earned"],
    )
    pd.testing.assert_frame_equal(
        student_term_data_hist, nsc_student_term_hist_data, check_like=True
    )


def test_calculate_pct_terms_or_courses_hist_course():
    test_df = pd.DataFrame(
        {
            "total_n_courses_enrolled_to_date": [2, 2, 10],
            "total_n_courses_Subject_MTH_to_date": [1, 2, 3],
            "total_n_courses_Subject_ENG_to_date": [0, 0, 2],
            "total_n_courses_Grade_A_to_date": [2, 1, 0],
            "pct_courses_Subject_MTH_to_date": [0.5, 1.0, 0.3],
            "pct_courses_Subject_ENG_to_date": [0.0, 0.0, 0.2],
            "pct_courses_Grade_A_to_date": [1.0, 0.5, 0.0],
        }
    )
    result_df = calculate_pct_terms_or_courses_hist(test_df.iloc[:, :4], level="course")
    pd.testing.assert_frame_equal(result_df, test_df)


def test_calculate_pct_terms_or_courses_hist_term():
    test_df = pd.DataFrame(
        {
            "term_n": [1, 2, 4],
            "n_terms_Full-Time": [1, 1, 2],
            "n_terms_Part-Time": [0, 1, 1],
            "pct_terms_Full-Time": [1.0, 0.5, 0.5],
            "pct_terms_Part-Time": [0.0, 0.5, 0.25],
        }
    )
    result_df = calculate_pct_terms_or_courses_hist(test_df.iloc[:, :3], level="term")
    pd.testing.assert_frame_equal(result_df, test_df)


def test_calculate_pct_terms_or_courses_hist_invalid_level():
    with pytest.raises(Exception):
        calculate_pct_terms_or_courses_hist(pd.DataFrame(), "corse")


def test_add_cumulative_nunique_col():
    test_df = pd.DataFrame(
        {
            "student_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "sort_col": [1, 2, 3, 1, 3, 2, 1, 2, 3],
            "other_col": [0, 0, 0, 0, 0, 2, 1, 1, 1],
            "Major": [
                "Math",
                "Chem",
                "Bio",
                "Math",
                "Math",
                "Chem",
                "None",
                "None",
                "Something",
            ],
            "nunique_Major_to_date": [1.0, 2.0, 3.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0],
        }
    )
    result_df = add_cumulative_nunique_col(
        test_df.iloc[:, :-1],
        sort_cols=["sort_col"],
        groupby_cols=["student_id"],
        colname="Major",
    )
    pd.testing.assert_frame_equal(result_df, test_df, check_like=True)


def test_calculate_avg_credits_rolling(nsc_student_term_hist_data):
    orig_cols = nsc_student_term_hist_data.columns
    credit_col = "term_total_Number of Credits Earned"
    result_df = calculate_avg_credits_rolling(
        nsc_student_term_hist_data,
        date_col="Term End Date",
        student_id_col="Student ID",
        credit_col=credit_col,
        n_days=400,
    )
    pd.testing.assert_frame_equal(nsc_student_term_hist_data, result_df[orig_cols])
    assert list(result_df[f"total_{credit_col}_400D_to_date"]) == [8.0, 13.0, 2.0]
    assert list(result_df["n_terms_enrolled_400D_to_date"]) == [1, 2, 1]
    assert list(result_df[f"avg_{credit_col}_400D_to_date"]) == [8.0, 6.5, 2.0]
