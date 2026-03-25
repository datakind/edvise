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

MIN_UNIQUENESS_PCT = 0.10        # Tier 1: high-cardinality ID-like columns (anchor candidates)
MIN_DISCRIMINATOR_NUNIQUE = 2    # Tier 2: must have at least 2 distinct non-null values (not constant)
MAX_NULL_RATE_TIER1 = 0.05       # Tier 1: strict null rate
MAX_NULL_RATE_TIER2 = 0.30       # Tier 2: discriminators can be noisier
EARLY_STOP_UNIQUENESS = 0.995    # stop single-col search if a column is essentially unique
MAX_CANDIDATE_POOL = 8           # top N columns per tier fed into combination search
MAX_KEY_SIZE = 6                 # maximum compound key width
TOP_K_CANDIDATES = 5             # number of ranked candidate keys to return

HIGH_DUPLICATE_RATE_THRESHOLD = 0.50  # switch to sample-based profiling above this
SAMPLE_GROUP_SIZE = 500               # number of duplicate groups to sample

STRUCTURAL_THRESHOLD = 0.70      # concentration_score >= this → structural
NOISE_THRESHOLD = 0.30           # concentration_score <= this → noise (round to nearest between)

TEMPORAL_NAME_PATTERNS = re.compile(
    r"(^date|_date$|^term$|_term$|^year$|_year$|semester|quarter|cohort|period|session|admit_date|enrollment_date)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CandidateKey(BaseModel):
    columns: list[str]
    uniqueness_score: float = Field(..., description="Fraction of rows unique on this key")
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
    sampled: bool = Field(False, description="True if classification is based on sampled groups due to high duplicate rate")


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

def _score_column(col: pd.Series, n_rows: int) -> Optional[tuple[str, float]]:
    """
    Returns (tier, score) or None if ineligible.
    Tier 1 — high cardinality, low nulls: anchor candidates for any key.
    Tier 2 — low cardinality but non-constant, moderate nulls ok: discriminator components only.
    """
    null_rate = col.isnull().mean()
    n_unique = col.nunique()
    uniqueness = n_unique / n_rows
    dtype_boost = 0.1 if pd.api.types.is_string_dtype(col) or pd.api.types.is_integer_dtype(col) else 0.0

    # Tier 1: strict null rate + high uniqueness
    if null_rate <= MAX_NULL_RATE_TIER1 and uniqueness >= MIN_UNIQUENESS_PCT:
        return ("tier1", uniqueness + dtype_boost)

    # Tier 2: relaxed null rate + just needs to not be constant
    if null_rate <= MAX_NULL_RATE_TIER2 and n_unique >= MIN_DISCRIMINATOR_NUNIQUE:
        return ("tier2", uniqueness + dtype_boost)

    return None


def _detect_candidate_keys(df: pd.DataFrame) -> list[CandidateKey]:
    n_rows = len(df)
    logger.info("Detecting candidate keys across %d columns, %d rows", len(df.columns), n_rows)

    tier1, tier2 = [], []
    for col in df.columns:
        result = _score_column(df[col], n_rows)
        if result is None:
            logger.debug("  Ineligible: %s", col)
            continue
        tier, score = result
        if tier == "tier1":
            tier1.append((col, score))
            logger.debug("  Tier 1 (ID): %s (score=%.4f)", col, score)
        else:
            tier2.append((col, score))
            logger.debug("  Tier 2 (discriminator): %s (score=%.4f)", col, score)

    tier1.sort(key=lambda x: x[1], reverse=True)
    tier2.sort(key=lambda x: x[1], reverse=True)
    tier1_cols = [c for c, _ in tier1[:MAX_CANDIDATE_POOL]]
    tier2_cols = [c for c, _ in tier2[:MAX_CANDIDATE_POOL]]

    logger.info("Tier 1 pool: %s", tier1_cols)
    logger.info("Tier 2 pool: %s", tier2_cols)

    if not tier1_cols:
        logger.warning("No Tier 1 columns found — cannot generate compound key candidates")
        return []

    candidates = []
    best_uniqueness = 0.0

    def _uniqueness(cols: list[str]) -> float:
        if len(cols) == 1:
            return df[cols[0]].nunique() / n_rows
        # Concatenate per-column hashes into a single compound hash — fully vectorized
        compound = sum(
            pd.util.hash_pandas_object(df[c], index=False) * (31 ** i)
            for i, c in enumerate(cols)
        )
        return compound.nunique() / n_rows

    def _is_dominated(combo: tuple[str, ...], best_by_subset: set[frozenset]) -> bool:
        """True if any strict subset of this combo already achieved best_uniqueness."""
        for size in range(1, len(combo)):
            for sub in combinations(combo, size):
                if frozenset(sub) in best_by_subset:
                    return True
        return False

    # Single-column candidates from Tier 1 only
    best_by_subset: set[frozenset] = set()
    for col in tier1_cols:
        uniqueness = df[col].nunique() / n_rows
        null_rate = df[col].isnull().mean()
        candidates.append(CandidateKey(
            columns=[col],
            uniqueness_score=round(uniqueness, 4),
            null_rate=round(null_rate, 4),
            rank=0,
        ))
        if uniqueness > best_uniqueness:
            best_uniqueness = uniqueness
        if uniqueness >= EARLY_STOP_UNIQUENESS:
            best_by_subset.add(frozenset([col]))
            logger.info("  Early stop: %s achieves %.4f uniqueness", col, uniqueness)
            break

    all_pool = tier1_cols + tier2_cols
    early_stopped = False
    for size in range(2, min(MAX_KEY_SIZE, len(all_pool)) + 1):
        if early_stopped:
            break
        logger.info("  Evaluating size-%d combinations...", size)
        for combo in combinations(all_pool, size):
            if not any(c in tier1_cols for c in combo):
                continue
            if _is_dominated(combo, best_by_subset):
                logger.debug("  Skipping dominated combo: %s", combo)
                continue
            uniqueness = _uniqueness(list(combo))
            null_rate = max(df[col].isnull().mean() for col in combo)
            candidates.append(CandidateKey(
                columns=list(combo),
                uniqueness_score=round(uniqueness, 4),
                null_rate=round(null_rate, 4),
                rank=0,
            ))
            if uniqueness >= EARLY_STOP_UNIQUENESS:
                best_by_subset.add(frozenset(combo))
                best_uniqueness = uniqueness
                logger.info("  Early stop on compound key %s (%.4f) — pruning larger sizes", combo, uniqueness)
                early_stopped = True
                break
            # If we already have enough candidates at the current best uniqueness, stop
            best_so_far = max((c.uniqueness_score for c in candidates), default=0.0)
            at_best = sum(1 for c in candidates if c.uniqueness_score >= best_so_far)
            if at_best >= TOP_K_CANDIDATES and uniqueness < best_so_far:
                logger.info("  Have %d candidates at best uniqueness %.4f — stopping combination search", at_best, best_so_far)
                early_stopped = True
                break

    # Rank by uniqueness descending, tiebreak by key length ascending (simpler = better)
    candidates.sort(key=lambda c: (c.uniqueness_score, -len(c.columns)), reverse=True)
    top = candidates[:TOP_K_CANDIDATES]
    for i, c in enumerate(top):
        c.rank = i + 1
        logger.info("  Candidate #%d: %s (uniqueness=%.4f)", i + 1, c.columns, c.uniqueness_score)

    return top


# ---------------------------------------------------------------------------
# Duplicate group classification
# ---------------------------------------------------------------------------

def _classify_group(group: pd.DataFrame) -> str:
    if group.duplicated(keep=False).all():
        return "true_duplicate"
    for col in group.columns:
        if group[col].dropna().nunique() > 1:
            return "competing_values"
    return "null_shadow"


def _is_temporal(col: str, dtype) -> bool:
    """
    Temporal if name matches pattern AND either:
    - dtype is datetime, OR
    - dtype is string/object (term codes are rarely parsed as datetime)
    Numeric columns require datetime dtype to avoid false positives.
    """
    name_match = bool(TEMPORAL_NAME_PATTERNS.search(col))
    if not name_match:
        return False
    return (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )


def _build_conflict_profiles(
    competing_groups: list[pd.DataFrame],
    total_competing: int,
) -> list[ColumnConflictProfile]:
    if not competing_groups:
        return []
    profiles = []
    for col in competing_groups[0].columns:
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
    # Temporal requires the TOP conflicting column to be temporal — not just any in top 3
    temporal_signal = top.is_temporal

    if temporal_signal and concentration_score >= NOISE_THRESHOLD:
        subtype = "temporal"
        temporal_col = next(p.column for p in conflict_profiles if p.is_temporal)
        hint = (
            f"Conflicts concentrated in temporal column `{temporal_col}`. "
            f"Rows likely represent the same entity across different time periods. "
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
    dup_groups_df = df[dupes_mask].copy()
    duplicate_rate = len(dup_groups_df) / len(df)
    logger.info("  Duplicate rows: %d / %d (%.1f%%)", len(dup_groups_df), len(df), 100 * duplicate_rate)

    if len(dup_groups_df) == 0:
        group_stats = DuplicateGroupStats(
            total_rows=len(df), duplicate_rows=0, affected_entities=0,
            group_size_distribution={}, null_shadow_count=0,
            true_duplicate_count=0, competing_values_count=0,
        )
        return CandidateKeyProfile(
            candidate_key=candidate, group_stats=group_stats,
            classification=DuplicateClassification(subtype="null_shadow", temporal_signal=False),
        )

    size_dist = {
        int(k): int(v)
        for k, v in dup_groups_df.groupby(list(key_cols)).size().value_counts().items()
    }
    affected_entities = dup_groups_df[list(key_cols)].drop_duplicates().shape[0]

    # High duplicate rate — sample before vectorized classification
    if duplicate_rate >= HIGH_DUPLICATE_RATE_THRESHOLD:
        logger.warning(
            "  High duplicate rate (%.1f%%) — sampling %d groups",
            100 * duplicate_rate, SAMPLE_GROUP_SIZE,
        )
        all_keys = dup_groups_df[list(key_cols)].drop_duplicates()
        sampled_keys = all_keys.sample(min(SAMPLE_GROUP_SIZE, len(all_keys)), random_state=42)
        dup_groups_df = dup_groups_df.merge(sampled_keys, on=list(key_cols))
        non_key_cols = [c for c in dup_groups_df.columns if c not in key_cols]
        sampled = True
    else:
        sampled = False

    # --- Vectorized group classification ---

    # True duplicate: all rows in group are identical (hash nunique == 1)
    dup_groups_df["_hash"] = pd.util.hash_pandas_object(dup_groups_df[non_key_cols], index=False)
    hash_nunique = dup_groups_df.groupby(list(key_cols))["_hash"].nunique()
    true_dup_keys = set(hash_nunique[hash_nunique == 1].index.tolist())

    # Among non-true-duplicates: null_shadow if max nunique across non-key cols <= 1
    remaining = dup_groups_df[~dup_groups_df.set_index(list(key_cols)).index.isin(true_dup_keys)]
    if len(remaining) > 0:
        nunique_per_col = (
            remaining.groupby(list(key_cols))[non_key_cols]
            .apply(lambda g: g.apply(lambda c: c.dropna().nunique()))
            .max(axis=1)
        )
        null_shadow_keys = set(nunique_per_col[nunique_per_col <= 1].index.tolist())
        competing_keys = set(nunique_per_col[nunique_per_col > 1].index.tolist())
    else:
        null_shadow_keys, competing_keys = set(), set()

    null_shadow_count = len(null_shadow_keys)
    true_dup_count = len(true_dup_keys)
    competing_count = len(competing_keys)

    logger.info(
        "  Group classification — null_shadow: %d, true_duplicate: %d, competing_values: %d",
        null_shadow_count, true_dup_count, competing_count,
    )

    group_stats = DuplicateGroupStats(
        total_rows=len(df),
        duplicate_rows=len(df[df.duplicated(subset=key_cols, keep=False)]),
        affected_entities=affected_entities,
        group_size_distribution=size_dist,
        null_shadow_count=null_shadow_count,
        true_duplicate_count=true_dup_count,
        competing_values_count=competing_count,
    )

    if not competing_keys:
        subtype = "true_duplicate" if true_dup_count and not null_shadow_count else "null_shadow"
        classification = DuplicateClassification(subtype=subtype, temporal_signal=False, sampled=sampled)
    else:
        # Build conflict profiles from competing groups only
        competing_df = dup_groups_df[
            dup_groups_df.set_index(list(key_cols)).index.isin(competing_keys)
        ]
        competing_groups = [
            group[non_key_cols]
            for _, group in competing_df.groupby(list(key_cols), sort=False)
        ]
        conflict_profiles = _build_conflict_profiles(competing_groups, competing_count)
        classification = _classify_competing(conflict_profiles, competing_count)
        classification.sampled = sampled
        logger.info(
            "  Classification: %s (concentration=%.4f)",
            classification.subtype, classification.concentration_score or 0,
        )

    return CandidateKeyProfile(candidate_key=candidate, group_stats=group_stats, classification=classification)


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
    profiled_uniqueness: set[float] = set()
    for candidate in candidate_keys:
        if candidate.uniqueness_score in profiled_uniqueness:
            logger.info(
                "  Skipping profile for %s — uniqueness %.4f already profiled",
                candidate.columns, candidate.uniqueness_score,
            )
            continue
        profiled_uniqueness.add(candidate.uniqueness_score)
        profiles.append(_profile_against_key(df, candidate))

    logger.info("=== DuplicateProfiler complete ===")
    return DuplicateProfile(candidate_key_profiles=profiles)