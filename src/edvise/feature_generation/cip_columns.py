"""CIP-format detection and program-of-study column resolution for feature generation."""

from __future__ import annotations

import logging
import re

import pandas as pd

from .column_names import CourseInputColumns

LOGGER = logging.getLogger(__name__)

# Matches full CIP codes (e.g. 52.0301, 24.01) used by extract_short_cip_code.
CIP_LIKE_PATTERN = re.compile(r"^\d[\d.]+$")

DEFAULT_MIN_CIP_FRACTION = 0.05


def _has_col(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col) and col in df.columns


def is_cip_like_value(value: object) -> bool:
    """True when ``value`` looks like a numeric CIP code, not free text."""
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return CIP_LIKE_PATTERN.match(text) is not None


def column_cip_match_fraction(ser: pd.Series) -> float:
    """Share of non-null values in ``ser`` that match :func:`is_cip_like_value`."""
    non_null = ser.dropna()
    if non_null.empty:
        return 0.0
    return float(non_null.map(is_cip_like_value).mean())


def has_sufficient_cip_values(
    df: pd.DataFrame,
    col: str | None,
    *,
    min_fraction: float = DEFAULT_MIN_CIP_FRACTION,
    min_count: int = 1,
) -> bool:
    """True when ``col`` exists and enough non-null values look like CIP codes."""
    if not _has_col(df, col):
        return False
    ser = df[col].astype("string")
    non_null = ser.dropna()
    if len(non_null) < min_count:
        return False
    return column_cip_match_fraction(ser) >= min_fraction


def warn_non_cip_program_column(
    df: pd.DataFrame,
    col: str,
    *,
    role: str,
    logger: logging.Logger | None = None,
) -> None:
    """Log when a program column is populated but values are not CIP-like."""
    log = logger or LOGGER
    if col not in df.columns:
        return
    ser = df[col].astype("string")
    n_non_null = int(ser.notna().sum())
    if n_non_null == 0:
        return
    frac = column_cip_match_fraction(ser)
    if frac < DEFAULT_MIN_CIP_FRACTION:
        log.warning(
            "Column %r (%s) has %s populated values but only %.1f%% match CIP format; "
            "program-of-study features will be skipped.",
            col,
            role,
            n_non_null,
            frac * 100,
        )


def resolve_term_program_of_study_source(
    df: pd.DataFrame,
    cols: CourseInputColumns,
    *,
    logger: logging.Logger | None = None,
) -> str | None:
    """
    Physical course column to read for PDP-style ``term_program_of_study`` features.

    Checks ``term_program_of_study`` then ``term_declared_major``; each must contain
    a sufficient share of CIP-like values.
    """
    candidates: list[tuple[str, str | None]] = [
        ("term_program_of_study", cols.term_program_of_study),
        ("term_declared_major", cols.term_declared_major),
    ]
    for role, col in candidates:
        if not _has_col(df, col):
            continue
        if has_sufficient_cip_values(df, col):
            return col
        warn_non_cip_program_column(df, col, role=role, logger=logger)
    return None
