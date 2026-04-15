import pandas as pd
import pytest

from edvise.genai.mapping.identity_agent.term_normalization.term_datetime import (
    term_components_to_datetime,
    term_components_to_datetime_from_series,
)
from edvise.genai.mapping.schema_mapping_agent.transformation import utilities as tx_utils


@pytest.mark.parametrize(
    ("season", "expected"),
    [
        ("FALL", pd.Timestamp("2021-08-01")),
        ("WINTER", pd.Timestamp("2021-12-01")),
        ("SPRING", pd.Timestamp("2022-01-01")),
        ("SUMMER", pd.Timestamp("2022-05-01")),
    ],
)
def test_term_components_to_datetime_academic_year_split(season, expected):
    df = pd.DataFrame(
        {
            "_edvise_term_academic_year": ["2021-22"],
            "_edvise_term_season": [season],
        }
    )
    out = term_components_to_datetime(df)
    assert len(out) == 1
    assert out.iloc[0] == expected


def test_term_components_to_datetime_null_inputs_yield_nat():
    df_year_na = pd.DataFrame(
        {
            "_edvise_term_academic_year": [pd.NA],
            "_edvise_term_season": ["FALL"],
        }
    )
    assert pd.isna(term_components_to_datetime(df_year_na).iloc[0])

    df_season_na = pd.DataFrame(
        {
            "_edvise_term_academic_year": ["2021-22"],
            "_edvise_term_season": [pd.NA],
        }
    )
    assert pd.isna(term_components_to_datetime(df_season_na).iloc[0])


def test_term_components_to_datetime_unknown_season_is_nat():
    df = pd.DataFrame(
        {
            "_edvise_term_academic_year": ["2021-22"],
            "_edvise_term_season": ["AUTUMN"],
        }
    )
    assert pd.isna(term_components_to_datetime(df).iloc[0])


def test_term_components_to_datetime_strips_and_uppercases_season():
    df = pd.DataFrame(
        {
            "_edvise_term_academic_year": ["2021-22"],
            "_edvise_term_season": ["  fall  "],
        }
    )
    out = term_components_to_datetime(df)
    assert out.iloc[0] == pd.Timestamp("2021-08-01")


def test_term_components_to_datetime_custom_season_to_month():
    df = pd.DataFrame(
        {
            "_edvise_term_academic_year": ["2021-22"],
            "_edvise_term_season": ["FALL"],
        }
    )
    out = term_components_to_datetime(
        df, season_to_month={"FALL": 9, "WINTER": 1, "SPRING": 3, "SUMMER": 7}
    )
    assert out.iloc[0] == pd.Timestamp("2021-09-01")


def test_term_components_to_datetime_from_series_matches_dataframe_api():
    year = pd.Series(["2021-22"], dtype="string")
    season = pd.Series(["FALL"], dtype="string")
    df = pd.DataFrame(
        {"_edvise_term_academic_year": year, "_edvise_term_season": season}
    )
    expected = term_components_to_datetime(df)
    assert term_components_to_datetime_from_series(year, season).equals(expected)
    assert tx_utils.term_components_to_datetime(year, season).equals(expected)


def test_term_components_to_datetime_preserves_index():
    idx = pd.Index(["a", "b", "c"], name="row")
    df = pd.DataFrame(
        {
            "_edvise_term_academic_year": ["2021-22", None, "2020-21"],
            "_edvise_term_season": ["SPRING", "FALL", " fall "],
        },
        index=idx,
    )
    out = term_components_to_datetime(df)
    assert out.index.equals(idx)
    assert out.loc["a"] == pd.Timestamp("2022-01-01")
    assert pd.isna(out.loc["b"])
    assert out.loc["c"] == pd.Timestamp("2020-08-01")
