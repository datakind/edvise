import logging

import pandas as pd

import edvise.dataio.es_course_converters as es_course_converters
from edvise.dataio.es_course_converters import handle_missing_grades


def _unique_nulls(n: int, *, entry: bool = True) -> pd.DataFrame:
    years, terms = ("2020-21", "2021-22"), ("FALL", "SPRING")
    df = pd.DataFrame(
        {
            "learner_id": [str(i) for i in range(n)],
            "course_prefix": "ENG",
            "course_number": [str(100 + i) for i in range(n)],
            "academic_year": [years[i % 2] for i in range(n)],
            "academic_term": [terms[i % 2] for i in range(n)],
            "grade": pd.NA,
            "course_credits_earned": 0.0,
            "course_credits_attempted": 3.0,
        }
    )
    if entry:
        entry_years = ("2019-20", "2020-21")
        df["entry_year"] = [entry_years[i % 2] for i in range(n)]
        df["entry_term"] = [terms[i % 2] for i in range(n)]
    return df


def test_dup_null_earned_kept_as_m_fills_credits():
    df = pd.DataFrame(
        {
            "learner_id": ["1", "1"],
            "course_prefix": ["ENG", "ENG"],
            "course_number": ["101", "101-1"],
            "academic_year": ["2020-21", "2021-22"],
            "academic_term": ["FALL", "FALL"],
            "grade": [pd.NA, "A"],
            "course_credits_earned": [pd.NA, 3.0],
            "course_credits_attempted": [pd.NA, 3.0],
        }
    )
    out = handle_missing_grades(df)
    assert list(out.index) == [0, 1]
    assert out.loc[0, "grade"] == "M"
    assert float(out.loc[0, "course_credits_earned"]) == 0.0
    assert float(out.loc[0, "course_credits_attempted"]) == 0.0
    assert out.loc[1, "course_number"] == "101-1"  # suffix not stripped on store


def test_all_null_zero_earned_siblings_keep_first_only():
    df = pd.DataFrame(
        {
            "learner_id": ["1", "1"],
            "course_prefix": ["ENG", "ENG"],
            "course_number": ["101", "101"],
            "grade": [pd.NA, pd.NA],
            "course_credits_earned": [pd.NA, 0.0],
            "course_credits_attempted": [pd.NA, pd.NA],
        }
    )
    out = handle_missing_grades(df)
    assert list(out.index) == [0]
    assert out.loc[0, "grade"] == "M"
    assert float(out.loc[0, "course_credits_earned"]) == 0.0
    assert float(out.loc[0, "course_credits_attempted"]) == 0.0


def test_unique_null_grade_dropped():
    df = pd.DataFrame(
        {
            "learner_id": ["1"],
            "course_prefix": ["ENG"],
            "course_number": ["101"],
            "grade": [pd.NA],
            "course_credits_earned": [3.0],
            "course_credits_attempted": [pd.NA],
        }
    )
    assert handle_missing_grades(df).empty


def test_reproducible_same_input_same_output():
    df = pd.DataFrame(
        {
            "learner_id": ["1", "1", "2"],
            "course_prefix": ["ENG", "ENG", "MAT"],
            "course_number": ["101", "101-2", "200"],
            "grade": [pd.NA, "B", pd.NA],
            "course_credits_earned": [2.0, 3.0, 0.0],
            "course_credits_attempted": [pd.NA, 3.0, 3.0],
        }
    )
    pd.testing.assert_frame_equal(handle_missing_grades(df), handle_missing_grades(df))


def test_high_unique_drop_logs_error_and_term_breakdowns(monkeypatch, caplog):
    monkeypatch.setattr(es_course_converters, "_HIGH_DROP", 2)
    caplog.set_level(logging.INFO)
    assert handle_missing_grades(_unique_nulls(3)).empty
    assert "dropped 3 unique null-grade rows" in caplog.text
    assert "contact the school" in caplog.text
    assert "All entry_year / entry_term pairs with counts:" in caplog.text
    assert "All academic_year / academic_term pairs with counts:" in caplog.text
    assert "2019-20" in caplog.text and "FALL" in caplog.text
    assert "2020-21" in caplog.text and "SPRING" in caplog.text


def test_below_high_drop_skips_term_breakdown_logs(monkeypatch, caplog):
    monkeypatch.setattr(es_course_converters, "_HIGH_DROP", 5)
    caplog.set_level(logging.INFO)
    assert handle_missing_grades(_unique_nulls(2)).empty
    assert "contact the school" not in caplog.text
    assert "All entry_year / entry_term pairs with counts:" not in caplog.text
    assert "All academic_year / academic_term pairs with counts:" not in caplog.text


def test_high_unique_drop_without_entry_cols_still_logs_academic(monkeypatch, caplog):
    monkeypatch.setattr(es_course_converters, "_HIGH_DROP", 2)
    caplog.set_level(logging.INFO)
    assert handle_missing_grades(_unique_nulls(2, entry=False)).empty
    assert "contact the school" in caplog.text
    assert "Missing fields: 'entry_year' or 'entry_term'" in caplog.text
    assert "All academic_year / academic_term pairs with counts:" in caplog.text


def test_custom_key_column_names():
    df = pd.DataFrame(
        {
            "sid": ["1", "1"],
            "prefix": ["ENG", "ENG"],
            "number": ["101", "101-1"],
            "grade": [pd.NA, "A"],
            "course_credits_earned": [pd.NA, 3.0],
            "course_credits_attempted": [pd.NA, 3.0],
        }
    )
    out = handle_missing_grades(
        df,
        learner_id_col="sid",
        course_prefix_col="prefix",
        course_number_col="number",
    )
    assert list(out.index) == [0, 1]
    assert out.loc[0, "grade"] == "M"
    assert float(out.loc[0, "course_credits_earned"]) == 0.0
