"""Deterministic grain-key profiling and dedup impact simulation (shared across agents)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from edvise.genai.mapping.identity_agent.profiling.constants import (
    INDEX_COLUMN_PATTERNS,
)
from edvise.genai.mapping.shared.profiling.variance import (
    ColumnVarianceProfile,
    WithinGroupVarianceResult,
    compute_within_group_variance,
)

if TYPE_CHECKING:
    from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
        GrainContract,
    )

MEASURE_COLUMN_HINTS = re.compile(
    r"(?:credit|gpa|units?|hours?|score|grade|letter_grade|points?|amount|rate|pct|percent|count|attempted|earned)",
    re.IGNORECASE,
)


def is_measure_column(column: str) -> bool:
    """True when a column name looks like a grade/credit/measure field."""
    return bool(MEASURE_COLUMN_HINTS.search(str(column).strip()))


def prepare_profiling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-memory prep aligned with KeyProfiler and cleaning step 7: strip synthetic index
    columns and drop full-row duplicates.
    """
    work = df.copy()
    index_cols = [c for c in work.columns if INDEX_COLUMN_PATTERNS.match(c)]
    if index_cols:
        work = work.drop(columns=index_cols)
    return work.drop_duplicates().reset_index(drop=True)


def _uniqueness_score(key_cols: list[str], df: pd.DataFrame) -> float:
    n_rows = len(df)
    if n_rows == 0:
        return 1.0
    if len(key_cols) == 1:
        return float(df[key_cols[0]].nunique() / n_rows)
    compound = sum(
        pd.util.hash_pandas_object(df[c], index=False) * (31**i)
        for i, c in enumerate(key_cols)
    )
    return float(compound.nunique() / n_rows)


def _canonical_learner_column_for_frame(df: pd.DataFrame) -> str:
    if "learner_id" in df.columns:
        return "learner_id"
    return "student_id"


def _column_profile_to_jsonable(profile: ColumnVarianceProfile) -> dict[str, Any]:
    return {
        "column": profile.column,
        "pct_groups_with_variance": profile.pct_groups_with_variance,
        "sample_values": profile.sample_values,
    }


@dataclass(frozen=True)
class KeyProfileSummary:
    key_columns: list[str]
    uniqueness_score: float
    non_unique_rows: int
    affected_groups: int
    group_size_distribution: dict[int, int]
    column_profiles: list[ColumnVarianceProfile]
    sampled: bool

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "key_columns": self.key_columns,
            "uniqueness_score": self.uniqueness_score,
            "non_unique_rows": self.non_unique_rows,
            "affected_groups": self.affected_groups,
            "group_size_distribution": self.group_size_distribution,
            "top_column_profiles": [
                _column_profile_to_jsonable(p) for p in self.column_profiles[:10]
            ],
            "sampled": self.sampled,
        }


@dataclass(frozen=True)
class DedupImpact:
    rows_before: int
    rows_after: int
    rows_dropped: int
    strategy_applied: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_dropped": self.rows_dropped,
            "strategy_applied": self.strategy_applied,
        }


def profile_key(df: pd.DataFrame, key_cols: list[str]) -> KeyProfileSummary:
    """Profile uniqueness and within-group variance for ``key_cols`` on ``df``."""
    from edvise.genai.mapping.identity_agent.execution.contract_utilities import (
        resolve_grain_key_columns,
    )

    cols = list(df.columns)
    try:
        resolved_keys = resolve_grain_key_columns(key_cols, cols)
    except ValueError as e:
        raise ValueError(f"profile_key: {e}") from e
    variance = compute_within_group_variance(df, resolved_keys)
    return KeyProfileSummary(
        key_columns=list(resolved_keys),
        uniqueness_score=round(_uniqueness_score(resolved_keys, df), 4),
        non_unique_rows=variance.non_unique_rows,
        affected_groups=variance.affected_groups,
        group_size_distribution=variance.group_size_distribution,
        column_profiles=variance.column_profiles,
        sampled=variance.sampled,
    )


def categorical_variance_columns(
    variance: WithinGroupVarianceResult | KeyProfileSummary,
    *,
    min_pct: float = 0.0,
) -> list[str]:
    """Non-measure columns with variance in duplicate key groups."""
    profiles = (
        variance.column_profiles
        if isinstance(variance, KeyProfileSummary)
        else variance.column_profiles
    )
    return [
        p.column
        for p in profiles
        if p.pct_groups_with_variance > min_pct and not is_measure_column(p.column)
    ]


def simulate_dedup_impact(
    df: pd.DataFrame,
    contract: GrainContract,
    *,
    prepared: bool = False,
) -> DedupImpact:
    """
    Run ``apply_grain_dedup`` on a copy and return row counts before/after.

    Uses the same profiling-frame prep as KeyProfiler unless ``prepared=True``.
    """
    from edvise.genai.mapping.identity_agent.execution.contract_utilities import (
        apply_grain_dedup,
    )

    work = df if prepared else prepare_profiling_frame(df)
    rows_before = len(work)
    strategy = contract.dedup_policy.strategy
    canonical = _canonical_learner_column_for_frame(work)
    out = apply_grain_dedup(
        work,
        contract,
        canonical_learner_column=canonical,
    )
    rows_after = len(out)
    return DedupImpact(
        rows_before=rows_before,
        rows_after=rows_after,
        rows_dropped=rows_before - rows_after,
        strategy_applied=strategy,
    )


__all__ = [
    "DedupImpact",
    "KeyProfileSummary",
    "categorical_variance_columns",
    "is_measure_column",
    "prepare_profiling_frame",
    "profile_key",
    "simulate_dedup_impact",
]
