from __future__ import annotations

import re
from itertools import combinations
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

MIN_UNIQUENESS_PCT = 0.10        # column must have >10% unique values to be a key candidate
MAX_NULL_RATE = 0.05             # column must have <5% null rate to be a key candidate
EARLY_STOP_UNIQUENESS = 0.995   # stop searching if a combo achieves this uniqueness
MAX_CANDIDATE_POOL = 8          # top N eligible columns fed into combination search
MAX_KEY_SIZE = 6                # maximum compound key width
TOP_K_CANDIDATES = 5            # number of ranked candidates to return

STRUCTURAL_THRESHOLD = 0.70     # concentration_score >= this → structural
NOISE_THRESHOLD = 0.30          # concentration_score <= this → noise
                                # between the two → round to nearest

TEMPORAL_NAME_PATTERNS = re.compile(
    r"(date|term|year|semester|quarter|cohort|period|session|admit|enroll)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CandidateKey(BaseModel):
    columns: list[str]
    uniqueness_score: float = Field(..., description="Fraction of rows that are unique on this key")
    null_rate: float = Field(..., description="Max null rate across key columns")
    rank: int


class ColumnConflictProfile(BaseModel):
    column: str
    conflict_count: int = Field(..., description="Number of duplicate groups where this column has >1 distinct non-null value")
    conflict_fraction: float = Field(..., description="conflict_count / total_competing_groups")
    is_temporal: bool = Field(..., description="True if column has date dtype or matches temporal name heuristics")


class DuplicateGroupStats(BaseModel):
    total_rows: int
    duplicate_rows: int
    affected_entities: int
    group_size_distribution: dict[int, int] = Field(..., description="group_size → count of entities with that many rows")
    null_shadow_count: int
    true_duplicate_count: int
    competing_values_count: int


class DuplicateClassification(BaseModel):
    subtype: str = Field(..., description="null_shadow | true_duplicate | structural | noise | temporal | ambiguous")
    concentration_score: Optional[float] = Field(None, description="Fraction of competing groups explained by top conflicting column. None for non-competing subtypes.")
    temporal_signal: bool = False
    top_conflicting_columns: list[ColumnConflictProfile] = []
    business_rule_hint: Optional[str] = Field(None, description="Human-readable hint about what business rule is needed")


class DuplicateProfile(BaseModel):
    candidate_keys: list[CandidateKey] = Field(..., description="Ranked candidate keys by uniqueness score")
    group_stats: DuplicateGroupStats
    classification: DuplicateClassification


# ---------------------------------------------------------------------------
# Candidate key detection
# ---------------------------------------------------------------------------

def _score_column(col: pd.Series, n_rows: int) -> Optional[float]:
    """Returns a composite candidate score for a column, or None if ineligible."""
    null_rate = col.isnull().mean()
    if null_rate > MAX_NULL_RATE:
        return None
    uniqueness = col.nunique() / n_rows
    if uniqueness < MIN_UNIQUENESS_PCT:
        return None
    # Boost ID-like dtypes
    dtype_boost = 0.1 if pd.api.types.is_string_dtype(col) or pd.api.types.is_integer_dtype(col) else 0.0
    return uniqueness + dtype_boost


def _detect_candidate_keys(df: pd.DataFrame) -> list[CandidateKey]:
    n_rows = len(df)
    scored = []
    for col in df.columns:
        score = _score_column(df[col], n_rows)
        if score is not None:
            scored.append((col, score))

    # Sort by composite score descending, take top N for combination search
    scored.sort(key=lambda x: x[1], reverse=True)
    pool = [col for col, _ in scored[:MAX_CANDIDATE_POOL]]

    candidates = []
    for size in range(1, min(MAX_KEY_SIZE, len(pool)) + 1):
        for combo in combinations(pool, size):
            uniqueness = df.drop_duplicates(subset=list(combo)).shape[0] / n_rows
            null_rate = max(df[col].isnull().mean() for col in combo)
            candidates.append(CandidateKey(
                columns=list(combo),
                uniqueness_score=round(uniqueness, 4),
                null_rate=round(null_rate, 4),
                rank=0,  # assigned after sorting
            ))
            if uniqueness >= EARLY_STOP_UNIQUENESS and size == 1:
                # Only early-stop on single columns to avoid masking compound key insights
                break

    # Rank by uniqueness score, return top K
    candidates.sort(key=lambda c: c.uniqueness_score, reverse=True)
    for i, c in enumerate(candidates[:TOP_K_CANDIDATES]):
        c.rank = i + 1
    return candidates[:TOP_K_CANDIDATES]


# ---------------------------------------------------------------------------
# Duplicate group stats + classification
# ---------------------------------------------------------------------------

def _classify_group(group: pd.DataFrame) -> str:
    non_key_cols = group.columns.tolist()
    if group.duplicated(subset=non_key_cols, keep=False).all():
        return "true_duplicate"
    for col in non_key_cols:
        if group[col].dropna().nunique() > 1:
            return "competing_values"
    return "null_shadow"


def _is_temporal(col: str, dtype) -> bool:
    return pd.api.types.is_datetime64_any_dtype(dtype) or bool(TEMPORAL_NAME_PATTERNS.search(col))


def _build_conflict_profiles(
    competing_groups: list[pd.DataFrame],
    total_competing: int,
) -> list[ColumnConflictProfile]:
    if not competing_groups:
        return []
    all_cols = competing_groups[0].columns.tolist()
    profiles = []
    for col in all_cols:
        count = sum(1 for g in competing_groups if g[col].dropna().nunique() > 1)
        if count == 0:
            continue
        dtype = competing_groups[0][col].dtype
        profiles.append(ColumnConflictProfile(
            column=col,
            conflict_count=count,
            conflict_fraction=round(count / total_competing, 4),
            is_temporal=_is_temporal(col, dtype),
        ))
    profiles.sort(key=lambda p: p.conflict_count, reverse=True)
    return profiles


def _classify_competing(
    conflict_profiles: list[ColumnConflictProfile],
    total_competing: int,
) -> DuplicateClassification:
    if not conflict_profiles:
        return DuplicateClassification(subtype="competing_values", concentration_score=0.0)

    top = conflict_profiles[0]
    concentration_score = top.conflict_fraction
    temporal_signal = any(p.is_temporal for p in conflict_profiles[:3])

    # Temporal takes precedence
    if temporal_signal and concentration_score >= NOISE_THRESHOLD:
        subtype = "temporal"
        hint = (
            f"Conflicts concentrated in temporal column `{next(p.column for p in conflict_profiles if p.is_temporal)}`. "
            f"Rows may represent the same entity at different points in time. "
            f"Consider elevating a time dimension to the grain."
        )
    elif concentration_score >= STRUCTURAL_THRESHOLD:
        subtype = "structural"
        hint = (
            f"Conflicts concentrated in `{top.column}` ({top.conflict_fraction:.0%} of competing groups). "
            f"Likely a grain mismatch — institution may have provided data at a finer grain than expected. "
            f"Business rule needed: which value of `{top.column}` should be treated as canonical?"
        )
    elif concentration_score <= NOISE_THRESHOLD:
        subtype = "noise"
        hint = (
            f"Conflicts scattered across multiple columns with no dominant pattern "
            f"(top column `{top.column}` explains only {top.conflict_fraction:.0%} of groups). "
            f"Likely upstream data entry issues or system migration artifacts. "
            f"Recommend DataKind review before institution escalation."
        )
    else:
        # Round to nearest
        subtype = "structural" if concentration_score >= 0.5 else "noise"
        hint = (
            f"Ambiguous conflict pattern — rounded to `{subtype}`. "
            f"Top conflicting column `{top.column}` explains {top.conflict_fraction:.0%} of groups."
        )

    return DuplicateClassification(
        subtype=subtype,
        concentration_score=round(concentration_score, 4),
        temporal_signal=temporal_signal,
        top_conflicting_columns=conflict_profiles[:5],
        business_rule_hint=hint,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def profile_duplicates(df: pd.DataFrame, key_cols: list[str]) -> DuplicateProfile:
    """
    Profile duplicates in a DataFrame against a candidate key.

    Args:
        df: Raw institution DataFrame
        key_cols: Candidate key columns to group by

    Returns:
        DuplicateProfile with candidate keys, group stats, and classification
    """
    n_rows = len(df)
    non_key_cols = [c for c in df.columns if c not in key_cols]

    # Candidate key detection
    candidate_keys = _detect_candidate_keys(df)

    # Duplicate group classification
    dupes_mask = df.duplicated(subset=key_cols, keep=False)
    clean_singles = df[~dupes_mask]
    dup_groups_df = df[dupes_mask]

    null_shadow, true_dup, competing = [], [], []
    competing_groups = []

    for _, group in dup_groups_df.groupby(key_cols, sort=False):
        g = group[non_key_cols]
        label = _classify_group(g)
        if label == "null_shadow":
            null_shadow.append(group)
        elif label == "true_duplicate":
            true_dup.append(group)
        else:
            competing.append(group)
            competing_groups.append(g)

    # Group size distribution (across all duplicate entities)
    size_dist: dict[int, int] = (
        dup_groups_df.groupby(key_cols).size().value_counts().to_dict()
        if len(dup_groups_df) > 0 else {}
    )

    group_stats = DuplicateGroupStats(
        total_rows=n_rows,
        duplicate_rows=len(dup_groups_df),
        affected_entities=dup_groups_df[key_cols].drop_duplicates().shape[0],
        group_size_distribution={int(k): int(v) for k, v in size_dist.items()},
        null_shadow_count=len(set(g[key_cols[0]].iloc[0] for g in null_shadow)) if null_shadow else 0,
        true_duplicate_count=len(set(g[key_cols[0]].iloc[0] for g in true_dup)) if true_dup else 0,
        competing_values_count=len(competing),
    )

    # Classification
    if len(dup_groups_df) == 0:
        classification = DuplicateClassification(subtype="null_shadow", temporal_signal=False)
    elif not competing:
        dominant = "true_duplicate" if true_dup and not null_shadow else "null_shadow"
        classification = DuplicateClassification(subtype=dominant, temporal_signal=False)
    else:
        conflict_profiles = _build_conflict_profiles(competing_groups, len(competing))
        classification = _classify_competing(conflict_profiles, len(competing))

    return DuplicateProfile(
        candidate_keys=candidate_keys,
        group_stats=group_stats,
        classification=classification,
    )