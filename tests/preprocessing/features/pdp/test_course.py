import pandas as pd
import pytest

from edvise.feature_generation import constants, course


@pytest.mark.parametrize(
    ["df", "min_passing_grade", "course_level_pattern", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "academic_year": ["2020-21", "2020-21", "2021-22"],
                    "academic_term": ["FALL", "SPRING", "FALL"],
                    "course_prefix": ["MATH", "MATH", "PHYS"],
                    "course_number": ["101", "202", "303"],
                    "course_cip": ["40.02", "45.06", "03.01"],
                    "course_type": ["CU", "CU", "CC"],
                    "delivery_method": ["O", "H", "F"],
                    "grade": ["4", "1", "W"],
                }
            ).astype({"grade": "string"}),
            2.0,
            constants.DEFAULT_COURSE_LEVEL_PATTERN,
            pd.DataFrame(
                {
                    "academic_year": ["2020-21", "2020-21", "2021-22"],
                    "academic_term": ["FALL", "SPRING", "FALL"],
                    "course_prefix": ["MATH", "MATH", "PHYS"],
                    "course_number": ["101", "202", "303"],
                    "course_cip": ["40.02", "45.06", "03.01"],
                    "course_type": ["CU", "CU", "CC"],
                    "delivery_method": ["O", "H", "F"],
                    "grade": ["4", "1", "W"],
                    "course_id": ["MATH101", "MATH202", "PHYS303"],
                    "course_subject_area": ["40", "45", "03"],
                    "course_passed": [True, False, pd.NA],
                    "course_completed": [True, True, False],
                    "course_level": [1, 2, 3],
                    "course_grade_numeric": [4.0, 1.0, pd.NA],
                    "course_grade": ["A", "D", "W"],
                }
            ).astype({"course_passed": "boolean", "course_grade_numeric": "Float32"}),
        ),
    ],
)
def test_add_course_features(df, min_passing_grade, course_level_pattern, exp):
    obs = course.add_features(
        df,
        min_passing_grade=min_passing_grade,
        course_level_pattern=course_level_pattern,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "prefix_col", "number_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "course_prefix": ["MATH", "BIO", "PHYS", "X", "Y", "Z"],
                    "course_number": ["101", "202", "303A", "404AB", "99", "001"],
                }
            ),
            "course_prefix",
            "course_number",
            pd.Series(["MATH101", "BIO202", "PHYS303A", "X404AB", "Y99", "Z001"]),
        ),
    ],
)
def test_course_id(df, prefix_col, number_col, exp):
    obs = course.course_id(df, prefix_col=prefix_col, number_col=number_col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "min_passing_grade", "exp"],
    [
        (
            pd.DataFrame({"grade": ["4", "2", "P", "F", "0"]}, dtype="string"),
            "grade",
            1.0,
            pd.Series([True, True, True, False, False], dtype="boolean"),
        ),
        (
            pd.DataFrame(
                {"grade": ["4.0", "1.9", "P", "F", "2.1", "I", "W"]}, dtype="string"
            ),
            "grade",
            2.0,
            pd.Series([True, False, True, False, True, pd.NA, pd.NA], dtype="boolean"),
        ),
    ],
)
def test_course_passed(df, col, min_passing_grade, exp):
    obs = course.course_passed(df, col=col, min_passing_grade=min_passing_grade)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame(
                {"grade": ["4", "1", "P", "F", "0", "I", "W"]}, dtype="string"
            ),
            "grade",
            pd.Series([True, True, True, True, True, False, False]),
        ),
    ],
)
def test_course_completed(df, col, exp):
    obs = course.course_completed(df, col=col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "pattern", "exp"],
    [
        (
            pd.DataFrame(
                {"course_number": ["101", "202", "303A", "404AB", "404ABC", "99", "1"]}
            ),
            "course_number",
            r"^(?P<course_level>\d)\d{2}(?:[A-Z]{,2})?$",
            pd.Series([1, 2, 3, 4, pd.NA, pd.NA, pd.NA], dtype="Int32"),
        ),
    ],
)
def test_course_level(df, col, pattern, exp):
    obs = course.course_level(df, col=col, pattern=pattern)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"grade": ["4", "2", "P", "F", "1"]}, dtype="string"),
            "grade",
            pd.Series([4.0, 2.0, pd.NA, pd.NA, 1.0], dtype="Float32"),
        ),
        (
            pd.DataFrame({"grade": ["4.0", "2.5", "I", "W", "1.75"]}, dtype="string"),
            "grade",
            pd.Series([4.0, 2.5, pd.NA, pd.NA, 1.75], dtype="Float32"),
        ),
    ],
)
def test_course_grade_numeric(df, col, exp):
    obs = course.course_grade_numeric(df, col=col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grade_col", "grade_num_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "grade": ["4", "2", "P", "F", "1"],
                    "grade_num": [4.0, 2.0, pd.NA, pd.NA, 1.0],
                },
            ).astype({"grade": "string", "grade_num": "Float32"}),
            "grade",
            "grade_num",
            pd.Series(["A", "C", "P", "F", "D"], dtype="string"),
        ),
        (
            pd.DataFrame(
                {
                    "grade": ["4.0", "3.3", "I", "W", "2.7", "0.7", "0.0", "A"],
                    "grade_num": [4.0, 3.3, pd.NA, pd.NA, 2.7, 0.7, 0.0, pd.NA],
                }
            ).astype({"grade": "string", "grade_num": "Float32"}),
            "grade",
            "grade_num",
            pd.Series(["A", "B", "I", "W", "B", "D", "F", "AUDIT"], dtype="string"),
        ),
    ],
)
def test_course_grade(df, grade_col, grade_num_col, exp):
    obs = course.course_grade(df, grade_col=grade_col, grade_num_col=grade_num_col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


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
    assert course.extract_course_level_from_course_number(input) == output


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
        course.extract_course_level_from_course_number(invalid_course_number)


def test_convert_number_of_courses_to_term_flag_col():
    test_df = pd.DataFrame(
        {
            "term_n_courses_something_a": [1, 0, 5],
            "term_n_courses_something_b": [9, 1, 0],
            "ignore_col": [1, 2, 3],
            "taking_course_something_a_this_term": [1, 0, 1],
            "taking_course_something_b_this_term": [1, 1, 0],
        }
    )
    result_df = course.convert_number_of_courses_cols_to_term_flag_cols(
        test_df.iloc[:, :3], col_prefix="term_n_courses_", orig_col="something"
    )
    pd.testing.assert_frame_equal(result_df, test_df.iloc[:, 2:])
