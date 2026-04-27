"""Tests for :mod:`edvise.data_audit.es_cohort_credential_years`."""

import pandas as pd
import pytest

from edvise.data_audit import es_cohort_credential_years as y


def test_add_columns_without_matriculation_creates_all_na() -> None:
    df = pd.DataFrame({"learner_id": ["a"]})
    out = y.add_es_credential_year_columns(df)
    for c in (
        y._FIRST_YEAR_BACHELORS,
        y._FIRST_YEAR_ASSOCIATES,
        y._YEARS_LATEST_ASSOCIATES,
        y._FIRST_YEAR_CERT,
        y._YEARS_LATEST_CERT,
    ):
        assert c in out.columns
        assert out[c].isna().all()


def test_same_day_award_is_year_one() -> None:
    d = pd.Timestamp("2020-08-15")
    df = pd.DataFrame(
        {
            "matriculation_date": [d],
            "bachelors_degree_conferral_date": [d],
        }
    )
    out = y.add_es_credential_year_columns(df)
    assert int(out.loc[0, y._FIRST_YEAR_BACHELORS]) == 1


def test_pre_matriculation_award_is_null() -> None:
    m = pd.Timestamp("2020-01-01")
    b = pd.Timestamp("2019-01-01")
    df = pd.DataFrame(
        {
            "matriculation_date": [m],
            "bachelors_degree_conferral_date": [b],
        }
    )
    out = y.add_es_credential_year_columns(df)
    assert pd.isna(out.loc[0, y._FIRST_YEAR_BACHELORS])


def test_bachelors_bucket_400_days() -> None:
    m = pd.Timestamp("2020-01-01")
    a = m + pd.Timedelta(days=400)
    df = pd.DataFrame(
        {
            "matriculation_date": [m],
            "bachelors_degree_conferral_date": [a],
        }
    )
    out = y.add_es_credential_year_columns(df)
    assert int(out.loc[0, y._FIRST_YEAR_BACHELORS]) == 2


@pytest.mark.parametrize("extra_days,expected", [(2922, 7), (8 * 365, 7)])
def test_bachelors_caps_at_seven(extra_days: int, expected: int) -> None:
    m = pd.Timestamp("2015-01-01")
    a = m + pd.Timedelta(days=extra_days)
    df = pd.DataFrame(
        {
            "matriculation_date": [m],
            "bachelors_degree_conferral_date": [a],
        }
    )
    out = y.add_es_credential_year_columns(df)
    assert int(out.loc[0, y._FIRST_YEAR_BACHELORS]) == expected


def test_cert_first_uses_min_last_uses_max() -> None:
    m = pd.Timestamp("2018-09-01")
    c1 = pd.Timestamp("2019-06-01")
    c3 = pd.Timestamp("2021-06-01")
    df = pd.DataFrame(
        {
            "matriculation_date": [m],
            "certificate1_date": [c1],
            "certificate2_date": [pd.NaT],
            "certificate3_date": [c3],
        }
    )
    out = y.add_es_credential_year_columns(df)
    f = int(out.loc[0, y._FIRST_YEAR_CERT])
    l = int(out.loc[0, y._YEARS_LATEST_CERT])
    assert f < l, "first cert should be earlier year bucket than last"


def test_associates_first_and_latest_match_single_date() -> None:
    m = pd.Timestamp("2019-01-01")
    a = m + pd.Timedelta(days=200)
    df = pd.DataFrame(
        {
            "matriculation_date": [m],
            "associates_degree_conferral_date": [a],
        }
    )
    out = y.add_es_credential_year_columns(df)
    assert out.loc[0, y._FIRST_YEAR_ASSOCIATES] == out.loc[0, y._YEARS_LATEST_ASSOCIATES]
