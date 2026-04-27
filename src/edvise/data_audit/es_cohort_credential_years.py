"""
PDP-style credential year index columns (1–7) for Edvise cohort (learner) rows.

Derived from ``matriculation_date`` and conferral or certificate dates after those
columns are parsed in :func:`edvise.dataio.read.read_raw_es_cohort_data` .
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cohort start + awards (``read.py`` / :class:`RawEdviseStudentDataSchema`)
MATRICULATION_COL = "matriculation_date"
BACHELORS_COL = "bachelors_degree_conferral_date"
ASSOCIATES_COL = "associates_degree_conferral_date"
CERTIFICATE_COLS: tuple[str, ...] = (
    "certificate1_date",
    "certificate2_date",
    "certificate3_date",
)

_FIRST_YEAR_BACHELORS = "first_year_to_bachelors_at_cohort_inst"
_FIRST_YEAR_ASSOCIATES = "first_year_to_associates_at_cohort_inst"
_YEARS_LATEST_ASSOCIATES = "years_to_latest_associates_at_cohort_inst"
_FIRST_YEAR_CERT = "first_year_to_certificate_at_cohort_inst"
_YEARS_LATEST_CERT = "years_to_latest_certificate_at_cohort_inst"


def _year_bucket_from_matriculation_and_award(
    matric: pd.Series,
    award: pd.Series,
) -> pd.Series:
    """
    Map elapsed time from matriculation to one award to PDP-style year index 1–7.

    Uses ``(award - matric).days / 365.25`` and ``ceil(max(years, 1/365.25))`` so
    same-day and within-first-year completions map to 1, then caps at 7. Rows with
    missing matric/award, or award before matriculation, are null.
    """
    out = pd.Series(pd.NA, index=matric.index, dtype="Int8")
    m = pd.to_datetime(matric, errors="coerce")
    a = pd.to_datetime(award, errors="coerce")
    days = (a - m).dt.days
    valid = m.notna() & a.notna() & (days >= 0)
    if not valid.any():
        return out
    d = days[valid].astype("float64")
    years_float = d / 365.25
    buck = np.ceil(np.maximum(years_float, 1.0 / 365.25))
    buck = np.clip(buck, 1.0, 7.0).astype("int64")
    out.loc[valid] = buck
    return out


def _certificate_first_last_dates(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    cols = [c for c in CERTIFICATE_COLS if c in df.columns]
    if not cols:
        nat = pd.Series(
            pd.NaT, index=df.index, dtype="datetime64[ns]"
        )
        return nat, nat
    part = df[cols]
    first = part.min(axis=1, skipna=True)
    last = part.max(axis=1, skipna=True)
    return first, last


def add_es_credential_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``first_year_to_*`` and ``years_to_latest_*`` at cohort institution, aligned
    with PDP naming, from Edvise date fields. Always emits the five column names
    (all-null :class:`pandas.NA` when the corresponding source dates are missing).

    Does not set other-institution columns (Edvise extract has no comparable dates).
    """
    idx = df.index
    na_i8 = pd.Series(pd.NA, index=idx, dtype="Int8")
    all_target = (
        _FIRST_YEAR_BACHELORS,
        _FIRST_YEAR_ASSOCIATES,
        _YEARS_LATEST_ASSOCIATES,
        _FIRST_YEAR_CERT,
        _YEARS_LATEST_CERT,
    )
    if MATRICULATION_COL not in df.columns:
        return df.assign(**{c: na_i8 for c in all_target})

    m = df[MATRICULATION_COL]

    b = (
        _year_bucket_from_matriculation_and_award(m, df[BACHELORS_COL])
        if BACHELORS_COL in df.columns
        else na_i8
    )
    if ASSOCIATES_COL in df.columns:
        assoc_bucket = _year_bucket_from_matriculation_and_award(
            m, df[ASSOCIATES_COL]
        )
    else:
        assoc_bucket = na_i8

    first_c, last_c = _certificate_first_last_dates(df)
    return df.assign(
        **{
            _FIRST_YEAR_BACHELORS: b,
            _FIRST_YEAR_ASSOCIATES: assoc_bucket,
            _YEARS_LATEST_ASSOCIATES: assoc_bucket,
            _FIRST_YEAR_CERT: _year_bucket_from_matriculation_and_award(m, first_c),
            _YEARS_LATEST_CERT: _year_bucket_from_matriculation_and_award(m, last_c),
        }
    )
