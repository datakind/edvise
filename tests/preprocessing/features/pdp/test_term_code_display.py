import pandas as pd
import pandas.testing as pdt

from edvise.feature_generation.term_code_display import (
    academic_term_category_from_term_code_display,
    academic_year_from_term_code_display,
    calendar_year_from_term_code,
)


def test_calendar_year_from_term_code_default_encoding() -> None:
    assert calendar_year_from_term_code("1179") == 2017
    assert calendar_year_from_term_code("1182") == 2018


def test_academic_year_from_display_john_jay_shape() -> None:
    s = pd.Series(["1179 Fall", "1182 Spring", "1186 Summer", pd.NA])
    got = academic_year_from_term_code_display(s)
    exp = pd.Series(["2017-18", "2017-18", "2017-18", pd.NA], dtype="string")
    pdt.assert_series_equal(got, exp)


def test_academic_term_category_from_display() -> None:
    s = pd.Series(["1179 Fall", "1182 Spring", "1186 Summer"])
    got = academic_term_category_from_term_code_display(s)
    assert list(got) == ["FALL", "SPRING", "SUMMER"]
