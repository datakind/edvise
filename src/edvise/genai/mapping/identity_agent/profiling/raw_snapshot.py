from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from edvise.configs.custom import CleaningConfig

from .constants import (
    SAMPLE_VALUES_TOP_N,
    TERM_COLUMN_PATTERNS,
    UNIQUE_VALUES_MAX_CARDINALITY,
)

from .schemas import RawColumnProfile, RawTableProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_term_candidate(col_name: str) -> bool:
    """Heuristic match against known term column name patterns."""
    return bool(TERM_COLUMN_PATTERNS.search(str(col_name)))


def _effective_null_series_for_profiling(
    series: pd.Series,
    null_tokens: list[str],
    *,
    treat_empty_strings_as_null: bool,
) -> pd.Series:
    """
    Per-column analogue of :func:`~edvise.data_audit.custom_cleaning.clean_dataset`
    null-token and whitespace steps (before dtype work), for ``null_rate_including_tokens``.
    """
    s = series.replace(null_tokens, np.nan)
    if treat_empty_strings_as_null and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
    ):
        s = s.replace(r"^\s*$", pd.NA, regex=True)
    return s


def _profile_column(
    series: pd.Series,
    *,
    null_tokens: list[str],
    treat_empty_strings_as_null: bool,
) -> RawColumnProfile:
    name = series.name
    dtype = str(series.dtype)
    null_rate = round(float(series.isnull().mean()), 4)
    eff = _effective_null_series_for_profiling(
        series,
        null_tokens,
        treat_empty_strings_as_null=treat_empty_strings_as_null,
    )
    null_rate_including_tokens = round(float(eff.isnull().mean()), 4)
    unique_count = int(series.nunique())

    non_null = series.dropna()

    # unique_values: only when cardinality is low enough to be useful
    unique_values = None
    if unique_count <= UNIQUE_VALUES_MAX_CARDINALITY:
        unique_values = sorted(
            [str(v) for v in non_null.unique().tolist()],
            key=lambda x: x.lower(),
        )

    # sample_values: top N by frequency
    sample_values = [
        str(v) for v in non_null.value_counts().head(SAMPLE_VALUES_TOP_N).index.tolist()
    ]

    return RawColumnProfile(
        name=name,
        dtype=dtype,
        null_rate=null_rate,
        null_rate_including_tokens=null_rate_including_tokens,
        unique_count=unique_count,
        unique_values=unique_values,
        sample_values=sample_values,
        is_term_candidate=_is_term_candidate(name),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def profile_raw_table(
    df: pd.DataFrame,
    institution_id: str,
    dataset: str,
    *,
    cleaning: CleaningConfig | None = None,
) -> RawTableProfile:
    """
    Lightweight raw column profiler. Produces dtype, null rate, unique values,
    and sample values for every column in the DataFrame.

    Designed to be called once per table, before any cleaning or dedup.
    Output is consumed by:
      - IdentityAgent term stage (via raw_profile.term_candidates)
      - profile_candidate_keys (bundled into KeyProfileResult)

    ``null_rate_including_tokens`` uses ``cleaning.null_tokens`` and
    ``cleaning.treat_empty_strings_as_null`` when ``cleaning`` is set; otherwise
    defaults match :func:`~edvise.data_audit.custom_cleaning.clean_dataset` with
    no config (``["(Blank)"]`` and empty strings as null on object/string columns).

    Args:
        df: Raw institution DataFrame (pre-normalization, pre-dedup)
        institution_id: Institution identifier string
        dataset: Logical dataset name (e.g. "student", "course", "semester")
        cleaning: Optional school cleaning config (same semantics as ``clean_dataset``).

    Returns:
        RawTableProfile with per-column profiles and term candidate flags
    """
    null_tokens = list(cleaning.null_tokens) if cleaning else ["(Blank)"]
    treat_empty = (
        cleaning.treat_empty_strings_as_null if cleaning else True
    )
    logger.info(
        "=== RawProfiler start — %s/%s: %d rows, %d columns ===",
        institution_id,
        dataset,
        len(df),
        len(df.columns),
    )

    columns = []
    for col in df.columns:
        col_profile = _profile_column(
            df[col],
            null_tokens=null_tokens,
            treat_empty_strings_as_null=treat_empty,
        )
        columns.append(col_profile)
        if col_profile.is_term_candidate:
            logger.debug(
                "  Term candidate: %s (dtype=%s, unique=%d)",
                col,
                col_profile.dtype,
                col_profile.unique_count,
            )

    term_count = sum(1 for c in columns if c.is_term_candidate)
    logger.info("  %d term candidate columns identified", term_count)
    logger.info("=== RawProfiler complete ===")

    return RawTableProfile(
        institution_id=institution_id,
        dataset=dataset,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
    )
