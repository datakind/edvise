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

# Tier 1 — anchors: top columns by n_unique within this table (not % of row count).
TIER1_MIN_NUNIQUE_ABS = 50       # absolute floor for large tables; capped by n_rows below
MIN_DISCRIMINATOR_NUNIQUE = 2    # Tier 2: must have at least 2 distinct values (not constant)
MAX_NULL_RATE_TIER1 = 0.05       # Tier 1: strict null rate
MAX_NULL_RATE_TIER2 = 0.30       # Tier 2: discriminators can be noisier; also Tier 1 fallback null cap
EARLY_STOP_UNIQUENESS = 0.995    # stop search if a key is essentially unique
MAX_CANDIDATE_POOL = 8           # top N columns per tier fed into combination search
MAX_KEY_SIZE = 6                 # maximum compound key width
TOP_K_CANDIDATES = 10            # ranked candidate keys returned & profiled (agent context)
LARGE_TABLE_ROW_THRESHOLD = 500_000
MAX_KEY_SIZE_LARGE_TABLE = 4     # cap width on large tables to avoid combinatorial blowups
MAX_COMBINATION_EVALS = 3_000    # hard cap of combo uniqueness evaluations
MAX_COMBINATION_EVALS_LARGE_TABLE = 900
NEAR_BEST_STOP_DELTA = 0.003     # treat near-best uniqueness as "good enough" plateau

HIGH_DUPLICATE_RATE_THRESHOLD = 0.50  # duplicate row fraction (for logging / heuristics)
SAMPLE_GROUP_SIZE = 500               # max duplicate groups used for classification (always capped)
PROFILE_MAX_WORK_ROWS = 150_000     # cap rows merged for hashing / groupby classification

STRUCTURAL_THRESHOLD = 0.70      # concentration_score >= this → structural
NOISE_THRESHOLD = 0.30           # concentration_score <= this → noise

# Pandas often emits Unnamed: 0, Unnamed: 0.1, etc. when reading CSV with odd indices.
INDEX_COLUMN_PATTERNS = re.compile(
    r"^(unnamed:\s*[\d.]+|index|row_number|row_num|rownum|row_id|__index_level_\d+__)$",
    re.IGNORECASE,
)

TEMPORAL_NAME_PATTERNS = re.compile(
    r"(?:^date$|_date$|^term$|_term$|^year$|_year$|semester|quarter|cohort|period|session|admit_date|enrollment_date)",
    re.IGNORECASE,
)

# Columns that typically anchor person-level identity in SIS-style files.
STUDENT_ANCHOR_NAME_PATTERN = re.compile(
    r"(?:student[_\s]?id|learner[_\s]?id|person[_\s]?id|enrollment[_\s]?id|sis[_\s]?id|"
    r"member[_\s]?id|participant[_\s]?id)",
    re.IGNORECASE,
)


def _student_anchor_column_names(columns: pd.Index) -> frozenset[str]:
    return frozenset(c for c in columns if STUDENT_ANCHOR_NAME_PATTERN.search(str(c)))


def _include_column_for_key_detection(series: pd.Series) -> bool:
    """
    Exclude dtypes that are almost never row-grain identifiers.
    Floats are usually measures (GPA, rates, imputations); int/string/object
    stay eligible so int64 IDs and coded keys remain in the pools.
    """
    return not pd.api.types.is_float_dtype(series)


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
    subtype: str = Field(
        ...,
        description=(
            "null_shadow | true_duplicate | structural | noise | temporal | "
            "grain_under_specified | competing_values"
        ),
    )
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

def _effective_tier1_nunique_floor(n_rows: int) -> int:
    """At least MIN_DISCRIMINATOR_NUNIQUE; at most min(TIER1_MIN_NUNIQUE_ABS, n_rows)."""
    if n_rows <= 0:
        return MIN_DISCRIMINATOR_NUNIQUE
    return max(MIN_DISCRIMINATOR_NUNIQUE, min(TIER1_MIN_NUNIQUE_ABS, n_rows))


def _column_stats(df: pd.DataFrame) -> list[tuple[str, int, float, bool]]:
    """(column_name, n_unique, null_rate, include_in_key_pools) per column, frame order."""
    rows = []
    for col in df.columns:
        series = df[col]
        null_rate = float(series.isnull().mean())
        n_unique = int(series.nunique())
        key_ok = _include_column_for_key_detection(series)
        if not key_ok:
            logger.debug("  Excluding from key pools (float dtype): %s", col)
        rows.append((col, n_unique, null_rate, key_ok))
    return rows


def _rank_tier1_pool(
    stats: list[tuple[str, int, float, bool]],
    n_rows: int,
    max_null_rate: float,
) -> list[str]:
    floor = _effective_tier1_nunique_floor(n_rows)
    eligible = [
        (c, u, nr)
        for c, u, nr, key_ok in stats
        if key_ok and nr <= max_null_rate and u >= floor
    ]
    eligible.sort(key=lambda t: (-t[1], t[0]))
    return [c for c, _, _ in eligible[:MAX_CANDIDATE_POOL]]


def _rank_tier2_pool(
    stats: list[tuple[str, int, float, bool]],
    tier1_cols: list[str],
    n_rows: int,
) -> list[str]:
    tier1_set = set(tier1_cols)
    eligible = [
        (c, u, nr)
        for c, u, nr, key_ok in stats
        if key_ok and c not in tier1_set and nr <= MAX_NULL_RATE_TIER2 and u >= MIN_DISCRIMINATOR_NUNIQUE
    ]
    eligible.sort(key=lambda t: (-t[1], t[0]))
    return [c for c, _, _ in eligible[:MAX_CANDIDATE_POOL]]


def _detect_candidate_keys(df: pd.DataFrame) -> list[CandidateKey]:
    n_rows = len(df)
    logger.info("Detecting candidate keys across %d columns, %d rows", len(df.columns), n_rows)

    stats = _column_stats(df)
    tier1_cols = _rank_tier1_pool(stats, n_rows, MAX_NULL_RATE_TIER1)
    if not tier1_cols:
        tier1_cols = _rank_tier1_pool(stats, n_rows, MAX_NULL_RATE_TIER2)
        if tier1_cols:
            logger.warning(
                "No Tier 1 columns under strict null rate (%.0f%%) — using looser cap (%.0f%%)",
                100 * MAX_NULL_RATE_TIER1,
                100 * MAX_NULL_RATE_TIER2,
            )
    tier2_cols = _rank_tier2_pool(stats, tier1_cols, n_rows)

    for c, u, nr, key_ok in stats:
        if not key_ok:
            continue
        if c in tier1_cols:
            logger.debug("  Tier 1 (anchor): %s n_unique=%d null_rate=%.4f", c, u, nr)
        elif c in tier2_cols:
            logger.debug("  Tier 2 (discriminator): %s n_unique=%d null_rate=%.4f", c, u, nr)

    logger.info("Tier 1 pool (within-table rank by n_unique): %s", tier1_cols)
    logger.info("Tier 2 pool (within-table rank by n_unique): %s", tier2_cols)

    if not tier1_cols:
        logger.warning("No Tier 1 columns found — cannot generate compound key candidates")
        return []

    def _uniqueness(cols: list[str]) -> float:
        if len(cols) == 1:
            return df[cols[0]].nunique() / n_rows
        compound = sum(
            pd.util.hash_pandas_object(df[c], index=False) * (31 ** i)
            for i, c in enumerate(cols)
        )
        return compound.nunique() / n_rows

    def _is_dominated(combo: tuple[str, ...], best_by_subset: set[frozenset]) -> bool:
        for size in range(1, len(combo)):
            for sub in combinations(combo, size):
                if frozenset(sub) in best_by_subset:
                    return True
        return False

    candidates = []
    best_by_subset: set[frozenset] = set()

    # Single-column candidates from Tier 1 only (emit all; do not break after the first
    # near-unique column — otherwise a synthetic index or date column can skip student_id).
    for col in tier1_cols:
        uniqueness = df[col].nunique() / n_rows
        null_rate = df[col].isnull().mean()
        candidates.append(CandidateKey(
            columns=[col],
            uniqueness_score=round(uniqueness, 4),
            null_rate=round(null_rate, 4),
            rank=0,
        ))
        if uniqueness >= EARLY_STOP_UNIQUENESS:
            best_by_subset.add(frozenset([col]))
            logger.info("  Near-unique single column: %s (%.4f)", col, uniqueness)

    # Compound candidates: must include at least one Tier 1 column.
    # We do not add compound keys to best_by_subset: a near-unique pair like
    # (student_id, class_number) must not suppress (student_id, class_number, term)
    # for agents reasoning about grain. best_by_subset is only for single-column
    # near-unique anchors from the loop above.
    all_pool = tier1_cols + tier2_cols
    effective_max_key_size = MAX_KEY_SIZE
    max_combination_evals = MAX_COMBINATION_EVALS
    if n_rows >= LARGE_TABLE_ROW_THRESHOLD:
        effective_max_key_size = min(MAX_KEY_SIZE, MAX_KEY_SIZE_LARGE_TABLE)
        max_combination_evals = MAX_COMBINATION_EVALS_LARGE_TABLE
        logger.info(
            "Large table detected (%d rows) — limiting max key width to %d and combo evals to %d",
            n_rows,
            effective_max_key_size,
            max_combination_evals,
        )
    stop_enumeration = False
    evaluated_combos = 0
    for size in range(2, min(effective_max_key_size, len(all_pool)) + 1):
        if stop_enumeration:
            break
        logger.info("  Evaluating size-%d combinations...", size)
        for combo in combinations(all_pool, size):
            if not any(c in tier1_cols for c in combo):
                continue
            if _is_dominated(combo, best_by_subset):
                logger.debug("  Skipping dominated combo: %s", combo)
                continue
            uniqueness = _uniqueness(list(combo))
            evaluated_combos += 1
            null_rate = max(df[col].isnull().mean() for col in combo)
            candidates.append(CandidateKey(
                columns=list(combo),
                uniqueness_score=round(uniqueness, 4),
                null_rate=round(null_rate, 4),
                rank=0,
            ))
            if uniqueness >= EARLY_STOP_UNIQUENESS:
                logger.info(
                    "  Near-unique compound key %s (%.4f) — continuing search for wider keys",
                    combo,
                    uniqueness,
                )
            if evaluated_combos >= max_combination_evals:
                logger.info(
                    "  Reached combo evaluation budget (%d) — stopping candidate search",
                    max_combination_evals,
                )
                stop_enumeration = True
                break
            # Prune enumeration only when we already have many top-tier keys and
            # this combo is strictly worse (does not block size-3 after a strong size-2).
            best_so_far = max((c.uniqueness_score for c in candidates), default=0.0)
            at_best = sum(1 for c in candidates if c.uniqueness_score >= best_so_far)
            if best_so_far >= EARLY_STOP_UNIQUENESS and at_best >= TOP_K_CANDIDATES:
                logger.info(
                    "  Have %d near-unique candidates at %.4f — stopping",
                    at_best,
                    best_so_far,
                )
                stop_enumeration = True
                break
            near_best = sum(
                1
                for c in candidates
                if c.uniqueness_score >= max(0.0, best_so_far - NEAR_BEST_STOP_DELTA)
            )
            if best_so_far >= EARLY_STOP_UNIQUENESS and near_best >= TOP_K_CANDIDATES * 2:
                logger.info(
                    "  Have %d near-best candidates within %.4f of best %.4f — stopping",
                    near_best,
                    NEAR_BEST_STOP_DELTA,
                    best_so_far,
                )
                stop_enumeration = True
                break
            if at_best >= TOP_K_CANDIDATES and uniqueness < best_so_far:
                logger.info("  Have %d candidates at best uniqueness %.4f — stopping", at_best, best_so_far)
                stop_enumeration = True
                break

    anchor_cols = _student_anchor_column_names(df.columns)
    if anchor_cols:
        before = len(candidates)
        candidates = [
            c
            for c in candidates
            if len(c.columns) == 1 or any(col in anchor_cols for col in c.columns)
        ]
        dropped = before - len(candidates)
        if dropped:
            logger.info(
                "  Dropped %d multi-column keys with no student-anchor column (identity context)",
                dropped,
            )

    # Uniqueness desc; prefer keys that include a student anchor when ties; shorter keys last tiebreak
    def _rank_key(c: CandidateKey) -> tuple[float, int, int]:
        has_anchor = 1 if (anchor_cols and any(col in anchor_cols for col in c.columns)) else 0
        if not anchor_cols:
            has_anchor = 1
        return (c.uniqueness_score, has_anchor, -len(c.columns))

    candidates.sort(key=_rank_key, reverse=True)
    top = candidates[:TOP_K_CANDIDATES]
    for i, c in enumerate(top):
        c.rank = i + 1
        logger.info("  Candidate #%d: %s (uniqueness=%.4f)", i + 1, c.columns, c.uniqueness_score)

    return top


# ---------------------------------------------------------------------------
# Conflict profiling + classification
# ---------------------------------------------------------------------------

def _is_temporal(col: str, dtype) -> bool:
    """
    Temporal if name matches pattern AND dtype is datetime or string/object.
    Numeric columns require datetime dtype to avoid false positives.
    """
    if not TEMPORAL_NAME_PATTERNS.search(col):
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
    # Temporal requires the TOP conflicting column to be temporal
    temporal_signal = top.is_temporal
    # Competing profiles use non-key columns only; a temporal top column means the key
    # omits a natural term/time dimension (e.g. course grain = student + class + term).
    if temporal_signal and concentration_score >= NOISE_THRESHOLD:
        subtype = "grain_under_specified"
        hint = (
            f"Duplicate groups differ on `{top.column}`, a time- or term-related field that is "
            f"not in this candidate key. If the business grain expects one row per student–course "
            f"offering **per term** (or similar), include `{top.column}` in the key — these rows are "
            f"often legitimate distinct enrollments, not erroneous duplicates."
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

def _profile_against_key(
    df: pd.DataFrame,
    candidate: CandidateKey,
    *,
    lightweight: bool = False,
) -> CandidateKeyProfile:
    key_cols = list(candidate.columns)
    non_key_cols = [c for c in df.columns if c not in key_cols]
    logger.info("Profiling duplicates against candidate key: %s", key_cols)

    # One groupby for global duplicate structure (no full duplicate-row materialization).
    sizes = df.groupby(key_cols, sort=False).size()
    dup_sizes = sizes[sizes > 1]

    if len(dup_sizes) == 0:
        group_stats = DuplicateGroupStats(
            total_rows=len(df), duplicate_rows=0, affected_entities=0,
            group_size_distribution={}, null_shadow_count=0,
            true_duplicate_count=0, competing_values_count=0,
        )
        return CandidateKeyProfile(
            candidate_key=candidate, group_stats=group_stats,
            classification=DuplicateClassification(subtype="null_shadow", temporal_signal=False),
        )

    duplicate_rows = int(dup_sizes.sum())
    duplicate_rate = duplicate_rows / len(df)
    affected_entities = len(dup_sizes)
    size_dist = {int(k): int(v) for k, v in dup_sizes.value_counts().items()}
    logger.info("  Duplicate rows: %d / %d (%.1f%%)", duplicate_rows, len(df), 100 * duplicate_rate)
    if duplicate_rate >= HIGH_DUPLICATE_RATE_THRESHOLD:
        logger.info(
            "  High duplicate row fraction (≥%.0f%%) — classification uses sampled groups/rows",
            100 * HIGH_DUPLICATE_RATE_THRESHOLD,
        )

    # Downsample duplicate keys / rows before hashing and per-group work.
    sampled = False
    keys_df = dup_sizes.index.to_frame(index=False)
    n_dup_groups = len(keys_df)
    if n_dup_groups > SAMPLE_GROUP_SIZE:
        keys_df = keys_df.sample(SAMPLE_GROUP_SIZE, random_state=42)
        sampled = True
        logger.info(
            "  Sampling %d duplicate groups (of %d) for classification",
            SAMPLE_GROUP_SIZE,
            n_dup_groups,
        )

    work = df.merge(keys_df, on=key_cols, how="inner")
    if len(work) > PROFILE_MAX_WORK_ROWS:
        work = work.sample(PROFILE_MAX_WORK_ROWS, random_state=43)
        sampled = True
        logger.info(
            "  Capped duplicate working rows at %d for classification",
            PROFILE_MAX_WORK_ROWS,
        )

    if not non_key_cols:
        group_stats = DuplicateGroupStats(
            total_rows=len(df),
            duplicate_rows=duplicate_rows,
            affected_entities=affected_entities,
            group_size_distribution=size_dist,
            null_shadow_count=0,
            true_duplicate_count=len(work.groupby(key_cols)),
            competing_values_count=0,
        )
        return CandidateKeyProfile(
            candidate_key=candidate,
            group_stats=group_stats,
            classification=DuplicateClassification(
                subtype="true_duplicate", temporal_signal=False, sampled=sampled,
            ),
        )

    # True duplicate: all rows in group are identical on non-key columns (hash nunique == 1)
    work["_hash"] = pd.util.hash_pandas_object(work[non_key_cols], index=False)
    hash_nunique = work.groupby(key_cols, sort=False)["_hash"].nunique()
    true_dup_keys = set(hash_nunique[hash_nunique == 1].index.tolist())

    remaining = work.loc[~work.set_index(key_cols).index.isin(true_dup_keys)].drop(
        columns=["_hash"], errors="ignore"
    )

    if len(remaining) > 0:
        nu_frames = [remaining.groupby(key_cols, sort=False)[c].nunique() for c in non_key_cols]
        nunique_per_col = pd.concat(nu_frames, axis=1).max(axis=1)
        null_shadow_keys = set(nunique_per_col[nunique_per_col <= 1].index.tolist())
        competing_keys = set(nunique_per_col[nunique_per_col > 1].index.tolist())
    else:
        null_shadow_keys, competing_keys = set(), set()

    work = work.drop(columns=["_hash"], errors="ignore")

    null_shadow_count = len(null_shadow_keys)
    true_dup_count = len(true_dup_keys)
    competing_count = len(competing_keys)

    logger.info(
        "  Group classification — null_shadow: %d, true_duplicate: %d, competing_values: %d",
        null_shadow_count, true_dup_count, competing_count,
    )

    group_stats = DuplicateGroupStats(
        total_rows=len(df),
        duplicate_rows=duplicate_rows,
        affected_entities=affected_entities,
        group_size_distribution=size_dist,
        null_shadow_count=null_shadow_count,
        true_duplicate_count=true_dup_count,
        competing_values_count=competing_count,
    )

    if not competing_keys:
        subtype = "true_duplicate" if true_dup_count and not null_shadow_count else "null_shadow"
        classification = DuplicateClassification(subtype=subtype, temporal_signal=False, sampled=sampled)
    elif lightweight:
        classification = _classify_competing([], competing_count)
        classification.sampled = sampled
        logger.info("  Classification (lightweight): %s", classification.subtype)
    else:
        competing_df = work.loc[work.set_index(key_cols).index.isin(competing_keys)]
        competing_groups = [
            group[list(non_key_cols)]
            for _, group in competing_df.groupby(key_cols, sort=False)
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

def profile_duplicates(df: pd.DataFrame, *, lightweight: bool = False) -> DuplicateProfile:
    """
    Self-contained duplicate profiler. Detects candidate keys and profiles
    duplicate patterns against each, returning a ranked DuplicateProfile
    for IdentityAgent to reason over.

    Args:
        df: Raw institution DataFrame
        lightweight: If True, skip per-column conflict profiling when rows compete on
            non-key fields (faster; subtype ``competing_values`` without hints).

    Returns:
        DuplicateProfile with per-candidate-key analysis
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== DuplicateProfiler start — %d rows, %d columns ===", len(df), len(df.columns))

    # Strip synthetic index columns before profiling
    index_cols = [c for c in df.columns if INDEX_COLUMN_PATTERNS.match(c)]
    if index_cols:
        logger.warning("Stripping synthetic index columns: %s", index_cols)
        df = df.drop(columns=index_cols)

    candidate_keys = _detect_candidate_keys(df)
    logger.info("Top %d candidate keys identified", len(candidate_keys))

    profiles = []
    for candidate in candidate_keys:
        profiles.append(_profile_against_key(df, candidate, lightweight=lightweight))

    logger.info("=== DuplicateProfiler complete ===")
    return DuplicateProfile(candidate_key_profiles=profiles)