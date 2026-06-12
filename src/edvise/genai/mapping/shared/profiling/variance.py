"""Deterministic within-group variance profiling for duplicate key groups (shared across agents)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

# Subset of profiling constants used for duplicate-group variance (also re-exported from
# ``identity_agent.profiling.constants`` for backward compatibility).
SAMPLE_GROUP_SIZE = 500
PROFILE_MAX_WORK_ROWS = 150_000
WITHIN_GROUP_SAMPLE_VALUES = 5


def _json_safe_profile_value(val: Any) -> Any:
    """Normalize a profiler sample value for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(val, "item"):
        try:
            return _json_safe_profile_value(val.item())
        except (ValueError, AttributeError):
            pass
    return str(val)


@dataclass(frozen=True)
class ColumnVarianceProfile:
    column: str
    pct_groups_with_variance: (
        float  # fraction of duplicate groups where this column varies
    )
    sample_values: (
        list  # up to WITHIN_GROUP_SAMPLE_VALUES unique values from the column
    )


@dataclass(frozen=True)
class WithinGroupVarianceResult:
    non_unique_rows: int
    affected_groups: int
    group_size_distribution: dict[int, int]  # group_size -> count_of_groups
    column_profiles: list[
        ColumnVarianceProfile
    ]  # sorted descending by pct_groups_with_variance
    sampled: bool


def compute_within_group_variance(
    df: pd.DataFrame,
    key_cols: list[str],
    profile_cols: list[str] | None = None,
) -> WithinGroupVarianceResult:
    """
    Profile duplicate groups on ``key_cols``: size distribution and per-column variance.

    When ``profile_cols`` is None, all columns except ``key_cols`` are profiled.
    """
    if profile_cols is None:
        cols_to_profile = [c for c in df.columns if c not in key_cols]
    else:
        cols_to_profile = [
            c for c in profile_cols if c in df.columns and c not in key_cols
        ]

    sizes = df.groupby(key_cols, sort=False).size()
    dup_sizes = sizes[sizes > 1]

    if len(dup_sizes) == 0:
        return WithinGroupVarianceResult(
            non_unique_rows=0,
            affected_groups=0,
            group_size_distribution={},
            column_profiles=[],
            sampled=False,
        )

    non_unique_rows = int(dup_sizes.sum())
    affected_groups = len(dup_sizes)
    size_dist = {int(k): int(v) for k, v in dup_sizes.value_counts().items()}

    sampled = False
    keys_df = dup_sizes.index.to_frame(index=False)
    if len(keys_df) > SAMPLE_GROUP_SIZE:
        keys_df = keys_df.sample(SAMPLE_GROUP_SIZE, random_state=42)
        sampled = True

    work = df.merge(keys_df, on=key_cols, how="inner")
    if len(work) > PROFILE_MAX_WORK_ROWS:
        work = work.sample(PROFILE_MAX_WORK_ROWS, random_state=43)
        sampled = True

    variance_profiles: list[ColumnVarianceProfile] = []
    for col in cols_to_profile:
        nunique = work.groupby(key_cols, sort=False)[col].nunique()
        pct_varying = round((nunique > 1).mean(), 4)
        sample_vals = [
            _json_safe_profile_value(v)
            for v in work[col].dropna().unique()[:WITHIN_GROUP_SAMPLE_VALUES]
        ]
        variance_profiles.append(
            ColumnVarianceProfile(
                column=col,
                pct_groups_with_variance=pct_varying,
                sample_values=sample_vals,
            )
        )

    variance_profiles.sort(key=lambda p: p.pct_groups_with_variance, reverse=True)

    return WithinGroupVarianceResult(
        non_unique_rows=non_unique_rows,
        affected_groups=affected_groups,
        group_size_distribution=size_dist,
        column_profiles=variance_profiles,
        sampled=sampled,
    )
