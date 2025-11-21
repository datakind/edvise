import pandas as pd
import numpy as np
import pytest

from edvise.feature_generation import term


@pytest.mark.parametrize(
    ["df", "first_term_of_year", "core_terms", "peak_covid_terms", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "123", "456", "456"],
                    "course_id": ["X101", "Y101", "X202", "Y202", "Z101", "Z202"],
                    "academic_year": [
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2019-20",
                        "2022-23",
                    ],
                    "academic_term": [
                        "FALL",
                        "FALL",
                        "WINTER",
                        "SPRING",
                        "FALL",
                        "SUMMER",
                    ],
                }
            ).astype(
                {
                    "academic_term": pd.CategoricalDtype(
                        ["FALL", "WINTER", "SPRING", "SUMMER"], ordered=True
                    )
                }
            ),
            "FALL",
            {"FALL", "SPRING"},
            {("2020-21", "SPRING"), ("2021-22", "FALL")},
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "123", "456", "456"],
                    "course_id": ["X101", "Y101", "X202", "Y202", "Z101", "Z202"],
                    "academic_year": [
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2019-20",
                        "2022-23",
                    ],
                    "academic_term": [
                        "FALL",
                        "FALL",
                        "WINTER",
                        "SPRING",
                        "FALL",
                        "SUMMER",
                    ],
                    "term_id": [
                        "2020-21 FALL",
                        "2020-21 FALL",
                        "2020-21 WINTER",
                        "2020-21 SPRING",
                        "2019-20 FALL",
                        "2022-23 SUMMER",
                    ],
                    "term_start_dt": pd.to_datetime(
                        [
                            "2020-09-01",
                            "2020-09-01",
                            "2021-01-01",
                            "2021-02-01",
                            "2019-09-01",
                            "2023-06-01",
                        ],
                    ),
                    "term_rank": [1, 1, 2, 3, 0, 4],
                    "term_rank_core": [1, 1, pd.NA, 2, 0, pd.NA],
                    "term_rank_noncore": [pd.NA, pd.NA, 0, pd.NA, pd.NA, 1],
                    "term_in_peak_covid": [False, False, False, True, False, False],
                    "term_is_core": [True, True, False, True, True, False],
                    "term_is_noncore": [False, False, True, False, False, True],
                }
            ).astype(
                {
                    "term_rank_core": "Int8",
                    "term_rank_noncore": "Int8",
                    "academic_term": pd.CategoricalDtype(
                        ["FALL", "WINTER", "SPRING", "SUMMER"], ordered=True
                    ),
                }
            ),
        ),
    ],
)
def test_add_term_features(df, first_term_of_year, core_terms, peak_covid_terms, exp):
    obs = term.add_features(
        df,
        first_term_of_year=first_term_of_year,
        core_terms=core_terms,
        peak_covid_terms=peak_covid_terms,
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp, check_dtype=False) is None


@pytest.mark.parametrize(
    ["df", "year_col", "term_col", "terms_subset", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year": ["22-23", "23-24", "22-23", "21-22", "22-23", None],
                    "term": ["FA", "FA", "SP", "SP", "FA", None],
                }
            ).astype({"term": pd.CategoricalDtype(["FA", "SP"], ordered=True)}),
            "year",
            "term",
            None,
            pd.Series([1, 3, 2, 0, 1, pd.NA], dtype="Int8"),
        ),
    ],
)
def test_term_rank(df, year_col, term_col, terms_subset, exp):
    obs = term.term_rank(
        df, year_col=year_col, term_col=term_col, terms_subset=terms_subset
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "year_col", "term_col", "peak_covid_terms", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year": ["20-21", "20-21", "21-22", "21-22", "21-22"],
                    "term": ["FA", "SP", "FA", "SP", "SU"],
                }
            ),
            "year",
            "term",
            [("20-21", "SP"), ("20-21", "SU"), ("21-22", "FA")],
            pd.Series([False, True, True, False, False], dtype="bool"),
        )
    ],
)
def test_term_in_peak_covid(df, year_col, term_col, peak_covid_terms, exp):
    obs = term.term_in_peak_covid(
        df, year_col=year_col, term_col=term_col, peak_covid_terms=peak_covid_terms
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "terms_subset", "term_col", "exp"],
    [
        (
            pd.DataFrame({"term": ["FALL", "WINTER", "SPRING", "SUMMER"]}),
            {"FALL", "SPRING"},
            "term",
            pd.Series([True, False, True, False], dtype="boolean"),
        ),
        (
            pd.DataFrame({"term": ["FALL", "WINTER", "SPRING", "SUMMER"]}),
            {"FALL", "WINTER", "SPRING"},
            "term",
            pd.Series([True, True, True, False], dtype="boolean"),
        ),
    ],
)
def test_term_in_subset(df, terms_subset, term_col, exp):
    obs = term.term_in_subset(df, terms_subset=terms_subset, term_col=term_col)
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp, check_names=False) is None


def test_extract_year_season():
    term_data = pd.DataFrame(
        {
            "term": ["2022SP", "2019S1", np.nan],
            0: [2022, 2019, np.nan],
            1: ["SP", "S1", np.nan],
        }
    )
    result = term.extract_year_season(term_data["term"])
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
        term.extract_year_season(invalid_df["term"])


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
    assert term.create_term_end_date(year, season) == pd.to_datetime(expected_date)


def test_create_term_end_date_raises_exception():
    with pytest.raises(Exception):
        term.create_term_end_date("2011-12", "Invalid Season")


def test_create_terms_lkp():
    max_year = 2015
    min_year = 2010
    possible_seasons = pd.DataFrame(
        {"season": ["Fall", "Winter", "Spring", "S1", "S2"], "order": [1, 2, 3, 4, 5]}
    )
    terms_lkp = term.create_terms_lkp(min_year, max_year, possible_seasons)
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


def test_add_term_order_with_extended_custom_season_map():
    # Custom season map adding two new "summer" subterms
    custom_map = {
        "spring": 1,
        "summer1": 2,
        "summer2": 3,
        "fall": 4,
        "winter": 5,
    }

    df = pd.DataFrame(
        {
            "term": [
                "Summer 1 2020",
                "Summer 2 2020",
                "Spring 2020",
                "Fall 2020",
            ]
        }
    )

    result = term.add_term_order(df, season_order_map=custom_map)

    # Extract season column (title-cased)
    assert list(result["season"]) == [
        "Summer1",
        "Summer2",
        "Spring",
        "Fall",
    ]

    # Season order values should follow the custom map
    expected_orders = {
        "Summer1": 2,
        "Summer2": 3,
        "Spring": 1,
        "Fall": 4,
    }
    actual_orders = dict(zip(result["season"], result["season_order"]))
    assert actual_orders == expected_orders

    # Term order uses the new mapping: year * 10 + season_order
    expected_term_order = {
        "Summer 1 2020": 2020 * 10 + 2,
        "Summer 2 2020": 2020 * 10 + 3,
        "Spring 2020":   2020 * 10 + 1,
        "Fall 2020":     2020 * 10 + 4,
    }
    actual_term_order = dict(zip(result["term"], result["term_order"]))
    assert actual_term_order == expected_term_order
