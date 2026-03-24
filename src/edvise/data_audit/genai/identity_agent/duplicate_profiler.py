from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

MIN_UNIQUENESS_PCT = 0.10
MAX_NULL_RATE = 0.05
EARLY_STOP_UNIQUENESS = 0.995
MAX_CANDIDATE_POOL = 8
MAX_KEY_SIZE = 6
TOP_K_CANDIDATES = 5

STRUCTURAL_THRESHOLD = 0.70
NOISE_THRESHOLD = 0.30

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
    conflict_count: int
    conflict_fraction: float
    is_temporal: bool


class DuplicateGroupStats(BaseModel):
    total_rows: int
    duplicate_rows: int
    affected_entities: int
    group_size_distribution: dict[int, int]
    null_shadow_count: int
    true_duplicate_count: int
    competing_values_count: int


class DuplicateClassification(BaseModel):
    subtype: str = Field(..., description="null_shadow | true_duplicate | structural | noise | temporal")
    concentration_score: Optional[float] = None
    temporal_signal: bool = False
    top_conflicting_columns: list[ColumnConflictProfile] = []
    business_rule_hint: Optional[str] = None


class CandidateKeyProfile(BaseModel):
    candidate_key: CandidateKey
    group_stats: DuplicateGroupStats
    classification: DuplicateClassification


class DuplicateProfile(BaseModel):
    candidate_key_profiles: list[CandidateKeyProfile] = Field(
        ..., description="Duplicate profile evaluated against each candidate key, ranked by uniqueness"
    )


# ---------------------------------------------------------------------------
# Candidate key detection
# ---------------------------------------------------------------------------

def _score_column(col: pd.Series, n_rows: int) -> Optional[float]:
    null_rate = col.isnull().mean()
    if null_rate > MAX_NULL_RATE:
        return None
    uniqueness = col.nunique() / n_rows
    if uniqueness < MIN_UNIQUENESS_PCT:
        return None
    dtype_boost = 0.1 if pd.api.types.is_string_dtype(col) or pd.api.types.is_integer_dtype(col) else 0.0
    return uniqueness + dtype_boost


def _detect_candidate_keys(df: pd.DataFrame) -> list[CandidateKey]:
    n_rows = len(df)
    logger.info("Detecting candidate keys across %d columns, %d rows", len(df.columns), n_rows)

    scored = []
    for col in df.columns:
        score = _score_column(df[col], n_rows)
        if score is not None:
            scored.append((col, score))
            logger.debug("  Column eligible: %s (score=%.4f)", col, score)
        else:
            logger.debug("  Column ineligible: %s", col)

    scored.sort(key=lambda x: x[1], reverse=True)
    pool = [col for col, _ in scored[:MAX_CANDIDATE_POOL]]
    logger.info("Candidate pool (top %d): %s", MAX_CANDIDATE_POOL, pool)

    candidates = []
    total_combos = sum(
        len(list(combinations(pool, size)))
        for size in range(1, min(MAX_KEY_SIZE, len(pool)) + 1)
    )
    logger.info("Evaluating up to %d key combinations (sizes 1-%d)", total_combos, min(MAX_KEY_SIZE, len(pool)))

    for size in range(1, min(MAX_KEY_SIZE, len(pool)) + 1):
        logger.info("  Evaluating size-%d combinations...", size)
        for combo in combinations(pool, size):
            uniqueness = df.drop_duplicates(subset=list(combo)).shape[0] / n_rows
            null_rate = max(df[col].isnull().mean() for col in combo)
            candidates.append(CandidateKey(
                columns=list(combo),
                uniqueness_score=round(uniqueness, 4),
                null_rate=round(null_rate, 4),
                rank=0,
            ))
            if size == 1 and uniqueness >= EARLY_STOP_UNIQUENESS:
                logger.info("  Early stop: single column %s achieves %.4f uniqueness", combo, uniqueness)
                break

    candidates.sort(key=lambda c: c.uniqueness_score, reverse=True)
    top = candidates[:TOP_K_CANDIDATES]
    for i, c in enumerate(top):
        c.rank = i + 1
        logger.info("  Candidate #%d: %s (uniqueness=%.4f)", i + 1, c.columns, c.uniqueness_score)

    return top


# ---------------------------------------------------------------------------
# Duplicate group stats + classification
# ---------------------------------------------------------------------------

def _classify_group(group: pd.DataFrame) -> str:
    if group.duplicated(keep=False).all():
        return "true_duplicate"
    for col in group.columns:
        if group[col].dropna().nunique() > 1:
            return "competing_values"
    return "null_shadow"


def _is_temporal(col: str, dtype) -> bool:
    """Temporal requires BOTH datetime dtype AND a matching name pattern."""
    return (
        pd.api.types.is_datetime64_any_dtype(dtype)
        and bool(TEMPORAL_NAME_PATTERNS.search(col))
    )


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

    if temporal_signal and concentration_score >= NOISE_THRESHOLD:
        subtype = "temporal"
        temporal_col = next(p.column for p in conflict_profiles if p.is_temporal)
        hint = (
            f"Conflicts concentrated in temporal column `{temporal_col}`. "
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
# Per-candidate profiling
# ---------------------------------------------------------------------------

def _profile_against_key(df: pd.DataFrame, candidate: CandidateKey) -> CandidateKeyProfile:
    key_cols = candidate.columns
    non_key_cols = [c for c in df.columns if c not in key_cols]
    logger.info("Profiling duplicates against candidate key: %s", key_cols)

    dupes_mask = df.duplicated(subset=key_cols, keep=False)
    dup_groups_df = df[dupes_mask]
    logger.info("  Duplicate rows: %d / %d (%.1f%%)", len(dup_groups_df), len(df), 100 * len(dup_groups_df) / len(df))

    null_shadow_ids, true_dup_ids, competing = [], [], []
    competing_groups = []

    for key_val, group in dup_groups_df.groupby(key_cols, sort=False):
        g = group[non_key_cols]
        label = _classify_group(g)
        if label == "null_shadow":
            null_shadow_ids.append(key_val)
        elif label == "true_duplicate":
            true_dup_ids.append(key_val)
        else:
            competing.append(group)
            competing_groups.append(g)

    logger.info(
        "  Group classification — null_shadow: %d, true_duplicate: %d, competing_values: %d",
        len(null_shadow_ids), len(true_dup_ids), len(competing),
    )

    size_dist: dict[int, int] = (
        {int(k): int(v) for k, v in dup_groups_df.groupby(key_cols).size().value_counts().items()}
        if len(dup_groups_df) > 0 else {}
    )

    group_stats = DuplicateGroupStats(
        total_rows=len(df),
        duplicate_rows=len(dup_groups_df),
        affected_entities=dup_groups_df[list(key_cols)].drop_duplicates().shape[0],
        group_size_distribution=size_dist,
        null_shadow_count=len(null_shadow_ids),
        true_duplicate_count=len(true_dup_ids),
        competing_values_count=len(competing),
    )

    if len(dup_groups_df) == 0:
        classification = DuplicateClassification(subtype="null_shadow", temporal_signal=False)
    elif not competing:
        dominant = "true_duplicate" if true_dup_ids and not null_shadow_ids else "null_shadow"
        classification = DuplicateClassification(subtype=dominant, temporal_signal=False)
    else:
        conflict_profiles = _build_conflict_profiles(competing_groups, len(competing))
        classification = _classify_competing(conflict_profiles, len(competing))
        logger.info("  Classification: %s (concentration=%.4f)", classification.subtype, classification.concentration_score or 0)

    return CandidateKeyProfile(
        candidate_key=candidate,
        group_stats=group_stats,
        classification=classification,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def profile_duplicates(df: pd.DataFrame) -> DuplicateProfile:
    """
    Self-contained duplicate profiler. Detects candidate keys and profiles
    duplicate patterns against each, returning a ranked DuplicateProfile
    for IdentityAgent to reason over.

    Args:
        df: Raw institution DataFrame

    Returns:
        DuplicateProfile with per-candidate-key analysis
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== DuplicateProfiler start — %d rows, %d columns ===", len(df), len(df.columns))

    candidate_keys = _detect_candidate_keys(df)
    logger.info("Top %d candidate keys identified", len(candidate_keys))

    profiles = []
    for candidate in candidate_keys:
        profile = _profile_against_key(df, candidate)
        profiles.append(profile)

    logger.info("=== DuplicateProfiler complete ===")
    return DuplicateProfile(candidate_key_profiles=profiles)