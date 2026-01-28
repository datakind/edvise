import functools as ft
import logging
import typing as t
from collections.abc import Collection

import pandas as pd
import numpy as np
import re

from edvise.utils import types
from . import constants, shared

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    *,
    first_term_of_year: types.TermType = constants.DEFAULT_FIRST_TERM_OF_YEAR,  # type: ignore
    core_terms: set[types.TermType] = constants.DEFAULT_CORE_TERMS,  # type: ignore
    peak_covid_terms: set[tuple[str, str]] = constants.DEFAULT_PEAK_COVID_TERMS,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
) -> pd.DataFrame:
    """
    Compute term-level features from pdp course dataset,
    and add as columns to ``df`` .

    Args:
        df
        first_term_of_year
        core_terms: Set of terms that together comprise the "core" of the academic year,
            in contrast with additional, usually shorter terms that may take place
            between core terms. Default value is {"FALL", "SPRING"}, which typically
            corresponds to a semester system; for schools on a trimester calendary,
            {"FALL", "WINTER", "SPRING"} is probably what you want.
        peak_covid_terms: Set of (year, term) pairs considered by the institution as
            occurring during "peak" COVID; for example, ``("2020-21", "SPRING")`` .
    """
    LOGGER.info("adding term features ...")
    noncore_terms: set[types.TermType] = set(df[term_col].unique()) - set(core_terms)
    df_term = (
        _get_unique_sorted_terms_df(df, year_col=year_col, term_col=term_col)
        # only need to compute features on unique terms, rather than at course-level
        # merging back into `df` afterwards ensures all rows have correct values
        .assign(
            term_id=ft.partial(shared.year_term, year_col=year_col, term_col=term_col),
            term_start_dt=ft.partial(
                shared.year_term_dt,
                col="term_id",
                bound="start",
                first_term_of_year=first_term_of_year,
            ),
            term_rank=ft.partial(term_rank, year_col=year_col, term_col=term_col),
            term_rank_core=ft.partial(
                term_rank,
                year_col=year_col,
                term_col=term_col,
                terms_subset=core_terms,
            ),
            term_rank_noncore=ft.partial(
                term_rank,
                year_col=year_col,
                term_col=term_col,
                terms_subset=noncore_terms,
            ),
            term_in_peak_covid=ft.partial(
                term_in_peak_covid,
                year_col=year_col,
                term_col=term_col,
                peak_covid_terms=peak_covid_terms,
            ),
            # yes, this is silly, but it helps a tricky feature computation later on
            term_is_core=ft.partial(
                term_in_subset, terms_subset=core_terms, term_col=term_col
            ),
            term_is_noncore=ft.partial(
                term_in_subset, terms_subset=noncore_terms, term_col=term_col
            ),
        )
    )
    return pd.merge(df, df_term, on=[year_col, term_col], how="inner")


def term_rank(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
    terms_subset: t.Optional[Collection[str]] = None,
) -> pd.Series:
    df_terms = (
        _get_unique_sorted_terms_df(df, year_col=year_col, term_col=term_col)
        if terms_subset is None
        else _get_unique_sorted_terms_df(
            df.loc[df[term_col].isin(terms_subset), :],
            year_col=year_col,
            term_col=term_col,
        )
    )
    df_terms_ranked = df_terms.assign(
        term_rank=lambda df: pd.Series(list(range(len(df))))
    )
    # left-join back into df, so this works if df rows are at the course *or* term level
    term_id_cols = [year_col, term_col]
    return (
        pd.merge(df[term_id_cols], df_terms_ranked, on=term_id_cols, how="left")
        .loc[:, "term_rank"]
        .rename(None)
        .astype("Int8")
    )


def _norm_token(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s)
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s or None


def add_term_order(
    df: pd.DataFrame,
    term_col: str = "term",
    season_order_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Season extraction is driven by `season_order_map` keys:

    - Keys are normalized to lowercase + single spaces.
    - For each term, we try to match the **longest** key that is a prefix of the
      normalized term string.
    - If nothing matches, we fall back to the first word (for default maps).
    """
    if term_col not in df.columns:
        raise KeyError(f"DataFrame must contain column '{term_col}'")

    g = df.copy()

    # safer string handling
    s = g[term_col].astype("string").str.strip()

    # year is still just "first 4-digit number in the string"
    year_str = s.str.extract(r"(\d{4})", expand=False)

    # Use default map if none provided
    if season_order_map is None:
        season_order_map = constants.DEFAULT_SEASON_ORDER_MAP

    # Normalize map keys
    norm_map: dict[str, int] = {}
    for k, v in season_order_map.items():
        nk = _norm_token(k)
        if nk is not None:
            norm_map[nk] = v

    # Precompute keys sorted by length (longest first) for prefix matching
    norm_keys = sorted(norm_map.keys(), key=len, reverse=True)

    def extract_season_token(term: str | None) -> str | None:
        if term is None:
            return None
        t_norm = _norm_token(term)
        if t_norm is None:
            return None

        # Try to match the longest prefix from season_order_map
        for key in norm_keys:
            if t_norm.startswith(key):
                return key

        # Fallback: first word (useful for default "Spring/Summer/Fall/Winter")
        first_word = t_norm.split(" ", 1)[0]
        if first_word in norm_map:
            return first_word

        return None

    # Vectorized-ish extraction via Series.apply (map size is tiny, so perf is fine)
    season_norm = s.apply(extract_season_token)

    found_seasons = set(season_norm.dropna().unique())
    valid_seasons = set(norm_map.keys())
    unexpected = found_seasons - valid_seasons
    if unexpected:
        LOGGER.warning(
            f"Unexpected seasons: {unexpected}. "
            f"Filtering to valid seasons: {valid_seasons}"
        )
        mask = season_norm.isin(valid_seasons)
        g = g[mask]
        season_norm = season_norm[mask]
        year_str = year_str[mask]

    # Pretty season for output: title-case + keep spaces & digits
    # e.g. "summer session 1" -> "Summer Session 1"
    g["season"] = season_norm.astype("string").str.title()

    # year as nullable Int64
    g["year"] = pd.to_numeric(year_str, errors="coerce").astype("Int64")

    # Map normalized season token -> order
    g["season_order"] = season_norm.map(norm_map).astype("Int64")

    # Core term definition is now up to you; simplest is still Spring/Fall:
    core_norm = {"spring", "fall"}
    g["is_core_term"] = season_norm.isin(core_norm)

    # Composite key
    g["term_order"] = (g["year"] * 10 + g["season_order"]).astype("Int64")

    return g


def term_in_peak_covid(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
    peak_covid_terms: set[tuple[str, str]],
) -> pd.Series:
    return pd.Series(
        pd.MultiIndex.from_frame(df[[year_col, term_col]]).isin(peak_covid_terms)
    )


def term_in_subset(
    df: pd.DataFrame, terms_subset: set[types.TermType], term_col: str = "academic_term"
) -> pd.Series:
    return df[term_col].isin(terms_subset).astype("boolean")


def create_terms_lkp(min_year, max_year, possible_seasons):
    """
    *CUSTOM SCHOOL FUNCTION*

    Create a dataframe of all possible terms.

    Args:
        min_year (str or int): earliest possible academic year
        max_year (str or int): latest possible academic year
        possible_seasons (pd.DataFrame): contains columns "season" and "order",
            where "season" indicates fall, spring, etc. in the format the school
            uses, and "order" is used to sort the seasons

    Returns:
        pd.DataFrame: all possible terms across the time frame, along with the rank order of each term,
            term_rank, the academic and calendar year, and the season
    """
    years = list(range(int(min_year), int(max_year) + 1))
    years = pd.DataFrame({"academic_year": [str(year) for year in years]})

    # doing this cross-join because one of our custom schools dropped the S2 term, but
    # for our definition of the outcome variable, we need each year to have the same number of terms
    terms_lkp = years.merge(possible_seasons, how="cross")
    terms_lkp["term_order"] = (
        terms_lkp["academic_year"] + terms_lkp["order"].astype(str)
    ).astype(int)
    # For one of our custom schools, term_order indicates the year of the fall of that academic year.
    # We define the calendar year as the next year for any season other than the Fall.
    terms_lkp["calendar_year"] = np.where(
        terms_lkp["season"] != "FA",
        terms_lkp["academic_year"].astype(int) + 1,
        terms_lkp["academic_year"],
    ).astype(int)
    terms_lkp["term"] = terms_lkp["calendar_year"].astype(str) + terms_lkp["season"]

    # The date created here itself is somewhat arbitrary
    # but can be used for windowing functions are better to look back 365 days rather
    # than x rows, etc.
    terms_lkp["term_end_date"] = terms_lkp[["season", "calendar_year"]].apply(
        lambda x: pd.to_datetime(
            str(_assign_month_to_season(x.season)) + "-01-" + str(x.calendar_year)
        ),
        axis=1,
    )

    terms_lkp["term_rank"] = terms_lkp["term_order"].rank(method="dense")
    return terms_lkp.sort_values("term_rank")


def create_term_end_date(academic_year, season):
    """
    *CUSTOM SCHOOL FUNCTION*

    Create term end date from an academic year and season

    Args:
        academic_year (str): school year in the format YYYY-YY
        season (str): Fall, Winter, Spring, or Summer

    Raises:
        Exception: if season is not one of the standard NSC seasons

    Returns:
        datetime
    """
    if season == "Fall":
        year = academic_year.split("-")[0]
    elif season in ["Winter", "Spring", "Summer"]:
        year = "20" + academic_year.split("-")[1]
    else:
        raise Exception(f"Invalid season {season}")

    month = _assign_month_to_season(season)

    return pd.to_datetime(f"{month}-01-{year}")


def extract_year_season(term_data):
    """
    *CUSTOM SCHOOL FUNCTION*

    Extract calendar year and season from term.

    Args:
        term_data (pd.Series): column of term data in the format YYYYTT, where
           YYYY is the calendar year, and TT denotes the term. For example, FA
           is Fall, SP is Spring, S1 and S2 are summer terms.

    Returns:
        pd.DataFrame: containing two columns - year and season
    """
    year_season_cols = term_data.str.extract(
        "^([0-9]{4})([a-zA-Z0-9]{2})$", expand=True
    )
    null_bool_index = year_season_cols.isna().any(axis=1)
    if (null_bool_index.sum() > 0) and (term_data[null_bool_index].notnull().sum() > 0):
        raise Exception(
            f"Term format not expected: {term_data[null_bool_index].unique()} Please revise the function and try again!"
        )
    year_season_cols[0] = pd.to_numeric(year_season_cols[0])
    return year_season_cols


# TODO: test
def _assign_month_to_season(season):
    """
    *CUSTOM SCHOOL FUNCTION*

    Assign a season to a month, for creating a datetime object.

    Args:
        season (str): season indicator, with possible values: Fall, FA,
            Winter, Spring, SP, S1, Summer, S2

    Raises:
        Exception: Season indicator not expected.

    Returns:
        int: month number of season
    """
    if season in ["Fall", "FA"]:
        return 12
    if season in ["Winter"]:
        return 2
    if season in ["Spring", "SP"]:
        return 6
    if season in ["S1"]:
        return 7
    if season in ["Summer", "S2"]:
        return 8
    else:
        raise Exception(f"Season {season} not expected. Try again!")


def _get_unique_sorted_terms_df(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
) -> pd.DataFrame:
    return (
        df[[year_col, term_col]]
        # dedupe more convenient op than groupby([year_col, term_col])
        .drop_duplicates(ignore_index=True)
        # null year and/or term values aren't relevant/useful here
        .dropna(axis="index", how="any")
        # assumes year col is alphanumerically sortable, term col is categorically ordered
        .sort_values(by=[year_col, term_col], ignore_index=True)
    )
