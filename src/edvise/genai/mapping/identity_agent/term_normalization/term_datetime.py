"""Combine IdentityAgent canonical term columns into calendar datetimes."""

from __future__ import annotations

import pandas as pd

_SEASON_TO_MONTH: dict[str, int] = {
    "SPRING": 1,
    "SUMMER": 5,
    "FALL": 8,
    "WINTER": 12,
}


def term_components_to_datetime(
    df: pd.DataFrame,
    year_col: str = "_edvise_term_academic_year",
    season_col: str = "_edvise_term_season",
    season_to_month: dict[str, int] | None = None,
) -> pd.Series:
    """
    Combine IdentityAgent canonical term columns into a datetime64[ns] Series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing IA-normalized term columns.
    year_col : str
        Column containing academic year strings in 'YYYY-YY' format
        (e.g. '2021-22'). FALL and WINTER use the first year; SPRING and
        SUMMER use the second year (reconstructed as first_year + 1).
    season_col : str
        Column containing canonical season strings
        (one of: SPRING, SUMMER, FALL, WINTER).
    season_to_month : dict[str, int], optional
        Override the default season-to-month mapping. Defaults to
        SPRING=1, SUMMER=5, FALL=8, WINTER=12.

    Returns
    -------
    pd.Series
        datetime64[ns] Series. Rows where either input is null produce NaT.
    """
    mapping = season_to_month or _SEASON_TO_MONTH

    first_year = (
        df[year_col].astype("string").str[:4].pipe(pd.to_numeric, errors="coerce")
    )
    season_upper = df[season_col].astype("string").str.strip().str.upper()
    month = season_upper.map(mapping)

    # FALL/WINTER → first year, SPRING/SUMMER → second year (first + 1)
    year = first_year.where(
        season_upper.isin({"FALL", "WINTER"}),
        other=first_year + 1,
    )

    valid = year.notna() & month.notna()

    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    result.loc[valid] = pd.to_datetime(
        {
            "year": year[valid].astype(int),
            "month": month[valid].astype(int),
            "day": 1,
        }
    )
    return result


def term_components_to_datetime_from_series(
    academic_year: pd.Series,
    season: pd.Series,
    *,
    season_to_month: dict[str, int] | None = None,
) -> pd.Series:
    """
    Same semantics as :func:`term_components_to_datetime` for aligned year and season Series.

    Used by SchemaMappingAgent ``utilities.term_components_to_datetime`` (registry) so
    implementation lives in one place.
    """
    df = pd.DataFrame(
        {
            "_edvise_term_academic_year": academic_year,
            "_edvise_term_season": season,
        }
    )
    return term_components_to_datetime(df, season_to_month=season_to_month)
