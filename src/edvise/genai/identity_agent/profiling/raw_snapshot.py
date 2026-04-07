from __future__ import annotations

import logging

import pandas as pd

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


def _profile_column(series: pd.Series) -> RawColumnProfile:
    name = series.name
    dtype = str(series.dtype)
    null_rate = round(float(series.isnull().mean()), 4)
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
) -> RawTableProfile:
    """
    Lightweight raw column profiler. Produces dtype, null rate, unique values,
    and sample values for every column in the DataFrame.

    Designed to be called once per table, before any cleaning or dedup.
    Output is consumed by:
      - IdentityAgent term config pass (via raw_profile.term_candidates)
      - profile_candidate_keys (bundled into KeyProfileResult)

    Args:
        df: Raw institution DataFrame (pre-normalization, pre-dedup)
        institution_id: Institution identifier string
        dataset: Logical dataset name (e.g. "student", "course", "semester")

    Returns:
        RawTableProfile with per-column profiles and term candidate flags
    """
    logger.info(
        "=== RawProfiler start — %s/%s: %d rows, %d columns ===",
        institution_id,
        dataset,
        len(df),
        len(df.columns),
    )

    columns = []
    for col in df.columns:
        col_profile = _profile_column(df[col])
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
