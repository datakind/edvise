import functools as ft
import re
import typing as t
from collections.abc import Sequence
from datetime import date

import pandas as pd

from edvise.utils import types, data_cleaning

from . import constants

RE_YEAR_TERM = re.compile(
    r"(?P<start_yr>\d{4})-(?P<end_yr>\d{2}) (?P<term>FALL|WINTER|SPRING|SUMMER)",
    flags=re.IGNORECASE,
)

TERM_BOUND_MONTH_DAYS = {
    "FALL": {"start": (9, 1), "end": (12, 31)},
    "WINTER": {"start": (1, 1), "end": (1, 31)},
    "SPRING": {"start": (2, 1), "end": (5, 31)},
    "SUMMER": {"start": (6, 1), "end": (8, 31)},
}


def extract_short_cip_code(ser: pd.Series) -> pd.Series:
    # NOTE: this simpler form works, but the values aren't nearly as clean
    # return ser.str.slice(stop=2).str.strip(".")
    return (
        ser.str.extract(r"^(?P<subject_area>\d[\d.])[\d.]+$", expand=False)
        .str.strip(".")
        .astype("string")
    )


def frac_credits_earned(
    df: pd.DataFrame,
    *,
    earned_col: str = "num_credits_earned",
    attempted_col: str = "num_credits_attempted",
) -> pd.Series:
    return df[earned_col].div(df[attempted_col])


def compute_values_equal(ser: pd.Series, to: t.Any | list[t.Any]) -> pd.Series:
    return ser.isin(to) if isinstance(to, list) else ser.eq(to)


def merge_many_dataframes(
    dfs: Sequence[pd.DataFrame],
    *,
    on: str | list[str],
    how: t.Literal["left", "right", "outer", "inner"] = "inner",
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge 2+ dataframes using the same set of merge parameters for each operation.

    See Also:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html
    """
    return ft.reduce(
        lambda left, right: pd.merge(left, right, on=on, how=how, sort=sort), dfs
    )


def year_term(df: pd.DataFrame, *, year_col: str, term_col: str) -> pd.Series:
    return df[year_col].str.cat(df[term_col], sep=" ")


def year_term_dt(
    df: pd.DataFrame,
    *,
    col: str,
    bound: t.Literal["start", "end"],
    first_term_of_year: types.TermType,
) -> pd.Series:
    """
    Compute an approximate start/end date for a given year-term,
    e.g. to compute time elapsed between course enrollments or to order course history.

    Args:
        df
        col: Column in ``df`` whose values represent academic/cohort year and term,
            formatted as "YYYY-YY TERM".
        bound: Which bound of the date range spanned by ``year_term`` to return;
            either the start (left) or end (right) bound.
        first_term_of_year: Term that officially begins the institution's academic year,
            either "FALL" or "SUMMER", which determines how the date's year is assigned.

    See Also:
        - :func:`year_term()`
    """
    return (
        df[col]
        .map(
            ft.partial(
                _year_term_to_dt, bound=bound, first_term_of_year=first_term_of_year
            ),
            na_action="ignore",
        )
        .astype("datetime64[s]")
    )


def _year_term_to_dt(
    year_term: str, bound: t.Literal["start", "end"], first_term_of_year: types.TermType
) -> date:
    if match := RE_YEAR_TERM.search(year_term):
        start_yr = int(match["start_yr"])
        term = match["term"].upper()
        yr = (
            start_yr
            if (term == "FALL" or (term == "SUMMER" and first_term_of_year == "SUMMER"))
            else start_yr + 1
        )
        mo, dy = TERM_BOUND_MONTH_DAYS[term][bound]
        return date(yr, mo, dy)
    else:
        raise ValueError(f"invalid year_term value: {year_term}")


def get_sum_hist_terms_or_courses_prefix(level):
    """
    *CUSTOM SCHOOL FUNCTION*

    Get prefix used to identify columns created using addition over time
    for either course- or term-level features.

    Args:
        level (str): course or term

    Raises:
        Exception: if level is not one of "course" or "term"

    Returns:
        str
    """
    if level == "course":
        return constants.TERM_COURSE_SUM_HIST_PREFIX
    if level == "term":
        return constants.TERM_FLAG_SUM_HIST_PREFIX
    else:
        raise Exception(f"Level {level} not expected. Try again!")


def get_n_terms_or_courses_col(level):
    """
    *CUSTOM SCHOOL FUNCTION*

    Get column name of a student's total courses over time or total terms over time

    Args:
        level (str): course or term

    Raises:
        Exception: if level is not one of "course" or "term"

    Returns:
        str
    """
    if level == "course":
        return (
            constants.TERM_COURSE_SUM_HIST_PREFIX + "enrolled" + constants.HIST_SUFFIX
        )
    if level == "term":
        return constants.TERM_NUMBER_COL
    else:
        raise Exception(f"Level {level} not expected. Try again!")


def get_sum_hist_terms_or_courses_cols(df, level):
    """
    *CUSTOM SCHOOL FUNCTION*

    Get column names from a dataframe created using addition over time
    for either course- or term-level features.

    Args:
        df (pd.DataFrame): contains column names prefixed by get_sum_hist_terms_or_courses_prefix()
            and the column get_n_terms_or_courses_col()
        level (str): course or term

    Returns:
        list[str]
    """
    orig_prefix = get_sum_hist_terms_or_courses_prefix(level)
    denominator_col = get_n_terms_or_courses_col(level)
    return [
        col
        for col in df.columns
        if col.startswith(orig_prefix) and col != denominator_col
    ]


def calculate_pct_terms_or_courses_hist(df, level):
    """
    *CUSTOM SCHOOL FUNCTION*

    Calculate percent of terms or courses with a particular characteristic
    across a student's history so far

    Args:
        df (pd.DataFrame): contains column names prefixed by get_sum_hist_terms_or_courses_prefix()
            and the column get_n_terms_or_courses_col()
        level (str): course or term

    Returns:
        pd.DataFrame: df with new percent of terms or courses columns
    """
    orig_prefix = get_sum_hist_terms_or_courses_prefix(level)
    denominator_col = get_n_terms_or_courses_col(level)
    numerator_cols = get_sum_hist_terms_or_courses_cols(df, level)

    # removes duplicates, keeps order
    numerator_cols = list(dict.fromkeys(numerator_cols))

    print(f"Calculating percent of {level}s to date for {len(numerator_cols)} columns")
    print(f"Sample of columns: {numerator_cols[:5]}")
    print(f"Denominator column: {denominator_col}")
    new_colnames = [
        col.replace(orig_prefix, f"pct_{level}s_") for col in numerator_cols
    ]
    df[new_colnames] = df.loc[:, numerator_cols].div(df[denominator_col], axis=0)

    # Convert only new feature columns those to snake_case
    df.rename(
        columns={col: data_cleaning.convert_to_snake_case(col) for col in new_colnames},
        inplace=True,
    )
    return df


def add_cumulative_nunique_col(df, sort_cols, groupby_cols, colname):
    """
    *CUSTOM SCHOOL FUNCTION*

    Calculate number of unique values within a group over time

    Args:
        df (pd.DataFrame): historical student data containing sort_cols, groupby_cols, and colname to count unique values
        sort_cols (list[str]): list of columns to sort by. For example, term or date.
        groupby_cols (list[str]): list of columns to group within. For example, Student ID.
        colname (str): column name to count unique values of over time

    Returns:
        pd.DataFrame: original data frame with new nunique_ column calculated over time.
    """
    sorted_df = df.sort_values(groupby_cols + sort_cols)
    new_colname = f"nunique_{colname}{constants.HIST_SUFFIX}"
    sorted_df[new_colname] = (
        sorted_df.drop_duplicates(groupby_cols + [colname])
        .groupby(groupby_cols)
        .cumcount()
        + 1
    )
    sorted_df[new_colname] = sorted_df[new_colname].ffill()
    return sorted_df
