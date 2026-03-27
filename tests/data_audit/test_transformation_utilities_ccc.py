"""Tests for CCC-oriented transformation utilities (academic year, term season, delimited suffix)."""

import pandas as pd
import pytest

from edvise.data_audit.genai.schema_mapping_agent.transformation import utilities as u


def test_format_academic_year_from_calendar_year_strings_and_ints():
    s = pd.Series(["2018", "2020", "1999", None])
    out = u.format_academic_year_from_calendar_year(s)
    expected = pd.Series(["2018-19", "2020-21", "1999-00", pd.NA], dtype="string")
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_format_academic_year_from_calendar_year_datetime():
    s = pd.to_datetime(pd.Series(["2018-08-17", "2019-01-14"]))
    out = u.format_academic_year_from_calendar_year(s)
    expected = pd.Series(["2018-19", "2019-20"], dtype="string")
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_term_season_from_datetime_bands():
    s = pd.to_datetime(
        pd.Series(["2018-08-17", "2019-01-14", "2020-06-05", "2021-07-01"])
    )
    out = u.term_season_from_datetime(s)
    expected = pd.Series(["FALL", "SPRING", "SUMMER", "SUMMER"], dtype="string")
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_term_season_from_datetime_nat():
    s = pd.Series([pd.NaT], dtype="datetime64[ns]")
    out = u.term_season_from_datetime(s)
    assert pd.isna(out.iloc[0])


def test_substring_after_first_delimiter_course_names():
    s = pd.Series(["ENGLISH-101", "BIOLOGY-121", "SPEECH-101-1", "NODELIM", None])
    out = u.substring_after_first_delimiter(s, delimiter="-")
    expected = pd.Series(["101", "121", "101-1", pd.NA, pd.NA], dtype="string")
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_substring_after_first_delimiter_empty_suffix():
    s = pd.Series(["ENGLISH-"])
    out = u.substring_after_first_delimiter(s)
    assert pd.isna(out.iloc[0])


def test_substring_after_first_delimiter_rejects_empty_delimiter():
    with pytest.raises(ValueError, match="non-empty"):
        u.substring_after_first_delimiter(pd.Series(["a"]), delimiter="")


def test_chained_extract_year_then_academic_year():
    raw = pd.to_datetime(pd.Series(["2018-09-01", "2020-01-15"]))
    y = u.extract_year(raw.astype("string"))
    out = u.format_academic_year_from_calendar_year(y)
    expected = pd.Series(["2018-19", "2020-21"], dtype="string")
    pd.testing.assert_series_equal(out, expected, check_names=False)
