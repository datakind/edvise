from __future__ import annotations

import logging
from itertools import combinations

import pandas as pd

from .constants import (
    EARLY_STOP_UNIQUENESS,
    INDEX_COLUMN_PATTERNS,
    LARGE_TABLE_KEY_SEARCH_SAMPLE_ROWS,
    LARGE_TABLE_ROW_THRESHOLD,
    MAX_CANDIDATE_POOL,
    MAX_COMBINATION_EVALS,
    MAX_COMBINATION_EVALS_LARGE_TABLE,
    MAX_KEY_SIZE,
    MAX_KEY_SIZE_LARGE_TABLE,
    MAX_NULL_RATE_TIER1,
    MAX_NULL_RATE_TIER2,
    MIN_DISCRIMINATOR_NUNIQUE,
    NEAR_BEST_STOP_DELTA,
    PROFILE_MAX_WORK_ROWS,
    SAMPLE_GROUP_SIZE,
    STUDENT_ANCHOR_NAME_PATTERN,
    TIER1_MIN_NUNIQUE_ABS,
    TOP_K_CANDIDATES,
    TOP_K_CANDIDATES_LARGE_TABLE,
    WITHIN_GROUP_SAMPLE_VALUES,
)
from .raw_snapshot import profile_raw_table
from .schemas import (
    CandidateKey,
    CandidateProfile,
    ColumnVarianceProfile,
    KeyProfileResult,
    RankedCandidateProfiles,
)

logger = logging.getLogger(__name__)


def _student_anchor_column_names(columns: pd.Index) -> frozenset[str]:
    return frozenset(c for c in columns if STUDENT_ANCHOR_NAME_PATTERN.search(str(c)))


def _include_column_for_key_detection(series: pd.Series) -> bool:
    """Exclude float columns — almost never row-grain identifiers (GPA, rates, etc.)."""
    return not pd.api.types.is_float_dtype(series)


# ---------------------------------------------------------------------------
# Candidate key detection — deterministic combinatorial work
# ---------------------------------------------------------------------------


def _effective_tier1_nunique_floor(n_rows: int) -> int:
    if n_rows <= 0:
        return MIN_DISCRIMINATOR_NUNIQUE
    return max(MIN_DISCRIMINATOR_NUNIQUE, min(TIER1_MIN_NUNIQUE_ABS, n_rows))


def _column_stats(df: pd.DataFrame) -> list[tuple[str, int, float, bool]]:
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
        if key_ok
        and c not in tier1_set
        and nr <= MAX_NULL_RATE_TIER2
        and u >= MIN_DISCRIMINATOR_NUNIQUE
    ]
    eligible.sort(key=lambda t: (-t[1], t[0]))
    return [c for c, _, _ in eligible[:MAX_CANDIDATE_POOL]]


def _detect_candidate_keys(df: pd.DataFrame) -> list[CandidateKey]:
    n_rows = len(df)
    logger.info(
        "Detecting candidate keys across %d columns, %d rows", len(df.columns), n_rows
    )
    top_k_limit = (
        TOP_K_CANDIDATES_LARGE_TABLE
        if n_rows >= LARGE_TABLE_ROW_THRESHOLD
        else TOP_K_CANDIDATES
    )

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

    for stat_col, u, nr, key_ok in stats:
        if not key_ok:
            continue
        if stat_col in tier1_cols:
            logger.debug(
                "  Tier 1 (anchor): %s n_unique=%d null_rate=%.4f", stat_col, u, nr
            )
        elif stat_col in tier2_cols:
            logger.debug(
                "  Tier 2 (discriminator): %s n_unique=%d null_rate=%.4f",
                stat_col,
                u,
                nr,
            )

    logger.info("Tier 1 pool: %s", tier1_cols)
    logger.info("Tier 2 pool: %s", tier2_cols)

    if not tier1_cols:
        logger.warning(
            "No Tier 1 columns found — cannot generate compound key candidates"
        )
        return []

    eval_df = df
    use_sampled_key_search = False
    if (
        n_rows >= LARGE_TABLE_ROW_THRESHOLD
        and len(df) > LARGE_TABLE_KEY_SEARCH_SAMPLE_ROWS
    ):
        eval_df = df.sample(LARGE_TABLE_KEY_SEARCH_SAMPLE_ROWS, random_state=44)
        use_sampled_key_search = True
        logger.info("Using %d-row sample for combo key search", len(eval_df))

    def _uniqueness(cols: list[str], source_df: pd.DataFrame) -> float:
        source_n_rows = len(source_df)
        if len(cols) == 1:
            return float(source_df[cols[0]].nunique() / source_n_rows)
        compound = sum(
            pd.util.hash_pandas_object(source_df[c], index=False) * (31**i)
            for i, c in enumerate(cols)
        )
        return float(compound.nunique() / source_n_rows)

    def _is_dominated(combo: tuple[str, ...], best_by_subset: set[frozenset]) -> bool:
        for size in range(1, len(combo)):
            for sub in combinations(combo, size):
                if frozenset(sub) in best_by_subset:
                    return True
        return False

    candidates: list[CandidateKey] = []
    best_by_subset: set[frozenset] = set()

    for col in tier1_cols:
        uniqueness = _uniqueness([col], eval_df)
        null_rate = df[col].isnull().mean()
        candidates.append(
            CandidateKey(
                columns=[col],
                uniqueness_score=round(uniqueness, 4),
                null_rate=round(null_rate, 4),
                rank=0,
            )
        )
        if uniqueness >= EARLY_STOP_UNIQUENESS:
            best_by_subset.add(frozenset([col]))
            logger.info("  Near-unique single column: %s (%.4f)", col, uniqueness)

    all_pool = tier1_cols + tier2_cols
    effective_max_key_size = (
        MAX_KEY_SIZE_LARGE_TABLE
        if n_rows >= LARGE_TABLE_ROW_THRESHOLD
        else MAX_KEY_SIZE
    )
    max_combination_evals = (
        MAX_COMBINATION_EVALS_LARGE_TABLE
        if n_rows >= LARGE_TABLE_ROW_THRESHOLD
        else MAX_COMBINATION_EVALS
    )

    stop_enumeration = False
    evaluated_combos = 0
    for size in range(2, min(effective_max_key_size, len(all_pool)) + 1):
        if stop_enumeration:
            break
        logger.info("  Evaluating size-%d combinations...", size)
        for combo in combinations(all_pool, size):
            if not any(col in tier1_cols for col in combo):
                continue
            if _is_dominated(combo, best_by_subset):
                continue
            uniqueness = _uniqueness(list(combo), eval_df)
            evaluated_combos += 1
            null_rate = max(df[col].isnull().mean() for col in combo)
            candidates.append(
                CandidateKey(
                    columns=list(combo),
                    uniqueness_score=round(uniqueness, 4),
                    null_rate=round(null_rate, 4),
                    rank=0,
                )
            )
            if uniqueness >= EARLY_STOP_UNIQUENESS:
                logger.info("  Near-unique compound key %s (%.4f)", combo, uniqueness)
            if evaluated_combos >= max_combination_evals:
                stop_enumeration = True
                break
            best_so_far = max(
                (ck.uniqueness_score for ck in candidates), default=0.0
            )
            at_best = sum(
                1 for ck in candidates if ck.uniqueness_score >= best_so_far
            )
            near_best = sum(
                1
                for ck in candidates
                if ck.uniqueness_score >= max(0.0, best_so_far - NEAR_BEST_STOP_DELTA)
            )
            if best_so_far >= EARLY_STOP_UNIQUENESS and at_best >= top_k_limit:
                stop_enumeration = True
                break
            if best_so_far >= EARLY_STOP_UNIQUENESS and near_best >= top_k_limit * 2:
                stop_enumeration = True
                break
            if at_best >= top_k_limit and uniqueness < best_so_far:
                stop_enumeration = True
                break

    anchor_cols = _student_anchor_column_names(df.columns)
    if anchor_cols:
        before = len(candidates)
        candidates = [
            ck
            for ck in candidates
            if len(ck.columns) == 1 or any(col in anchor_cols for col in ck.columns)
        ]
        dropped = before - len(candidates)
        if dropped:
            logger.info(
                "  Dropped %d multi-column keys with no student-anchor column", dropped
            )

    def _rank_key(c: CandidateKey) -> tuple[float, int, int]:
        has_anchor = (
            1 if (anchor_cols and any(col in anchor_cols for col in c.columns)) else 0
        )
        if not anchor_cols:
            has_anchor = 1
        return (c.uniqueness_score, has_anchor, -len(c.columns))

    if use_sampled_key_search and candidates:
        prelim_sorted = sorted(candidates, key=_rank_key, reverse=True)
        finalists = prelim_sorted[: max(2 * TOP_K_CANDIDATES, TOP_K_CANDIDATES)]
        for c in finalists:
            c.uniqueness_score = round(_uniqueness(c.columns, df), 4)
        candidates = finalists

    candidates.sort(key=_rank_key, reverse=True)
    top = candidates[:top_k_limit]
    for i, c in enumerate(top):
        c.rank = i + 1
        logger.info(
            "  Candidate #%d: %s (uniqueness=%.4f)",
            i + 1,
            c.columns,
            c.uniqueness_score,
        )

    return top


# ---------------------------------------------------------------------------
# Per-candidate key profiling — facts only, no interpretation
# ---------------------------------------------------------------------------


def _profile_against_key(df: pd.DataFrame, candidate: CandidateKey) -> CandidateProfile:
    key_cols = list(candidate.columns)
    non_key_cols = [c for c in df.columns if c not in key_cols]
    logger.info("Profiling against candidate key: %s", key_cols)

    sizes = df.groupby(key_cols, sort=False).size()
    dup_sizes = sizes[sizes > 1]

    if len(dup_sizes) == 0:
        return CandidateProfile(
            candidate_key=candidate,
            non_unique_rows=0,
            affected_groups=0,
            group_size_distribution={},
            within_group_variance=[],
            sampled=False,
        )

    non_unique_rows = int(dup_sizes.sum())
    affected_groups = len(dup_sizes)
    size_dist = {int(k): int(v) for k, v in dup_sizes.value_counts().items()}
    logger.info(
        "  Non-unique rows: %d / %d (%.1f%%)",
        non_unique_rows,
        len(df),
        100 * non_unique_rows / len(df),
    )

    sampled = False
    keys_df = dup_sizes.index.to_frame(index=False)
    if len(keys_df) > SAMPLE_GROUP_SIZE:
        keys_df = keys_df.sample(SAMPLE_GROUP_SIZE, random_state=42)
        sampled = True
        logger.info("  Sampling %d groups for variance profiling", SAMPLE_GROUP_SIZE)

    work = df.merge(keys_df, on=key_cols, how="inner")
    if len(work) > PROFILE_MAX_WORK_ROWS:
        work = work.sample(PROFILE_MAX_WORK_ROWS, random_state=43)
        sampled = True

    # Per non-key column: what fraction of non-unique groups have variance?
    # Raw signal for IdentityAgent to reason about grain — no interpretation here.
    variance_profiles = []
    for col in non_key_cols:
        nunique = work.groupby(key_cols, sort=False)[col].nunique()
        pct_varying = round((nunique > 1).mean(), 4)
        sample_vals = work[col].dropna().unique()[:WITHIN_GROUP_SAMPLE_VALUES].tolist()
        variance_profiles.append(
            ColumnVarianceProfile(
                column=col,
                pct_groups_with_variance=pct_varying,
                sample_values=sample_vals,
            )
        )

    variance_profiles.sort(key=lambda p: p.pct_groups_with_variance, reverse=True)

    return CandidateProfile(
        candidate_key=candidate,
        non_unique_rows=non_unique_rows,
        affected_groups=affected_groups,
        group_size_distribution=size_dist,
        within_group_variance=variance_profiles,
        sampled=sampled,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def profile_candidate_keys(
    df: pd.DataFrame,
    institution_id: str,
    dataset: str,
) -> KeyProfileResult:
    """
    Deterministic key profiler. Runs raw column profiling, then detects candidate
    keys and profiles each for group size distribution and per-column within-group
    variance.

    Interpretation (grain inference, dedup policy, cleaning hooks) is
    intentionally left to the IdentityAgent LLM call (Step 2). This function
    produces facts only.

    Args:
        df: Raw institution DataFrame (pre-normalization)
        institution_id: Institution identifier (passed through to raw table profile)
        dataset: Logical dataset name (e.g. ``student``, ``course``)

    Returns:
        KeyProfileResult with raw column stats and per-candidate-key stats
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger.info(
        "=== KeyProfiler start — %d rows, %d columns ===", len(df), len(df.columns)
    )

    index_cols = [c for c in df.columns if INDEX_COLUMN_PATTERNS.match(c)]
    if index_cols:
        logger.warning("Stripping synthetic index columns: %s", index_cols)
        df = df.drop(columns=index_cols)

    raw_table_profile = profile_raw_table(
        df, institution_id=institution_id, dataset=dataset
    )

    candidate_keys = _detect_candidate_keys(df)
    logger.info("Top %d candidate keys identified", len(candidate_keys))

    profiles = []
    unique_keys: list[frozenset[str]] = []
    for candidate in candidate_keys:
        candidate_set = frozenset(candidate.columns)
        if any(base.issubset(candidate_set) for base in unique_keys):
            logger.info(
                "  Skipping %s — superset of fully unique key", candidate.columns
            )
            profiles.append(
                CandidateProfile(
                    candidate_key=candidate,
                    non_unique_rows=0,
                    affected_groups=0,
                    group_size_distribution={},
                    within_group_variance=[],
                    sampled=False,
                )
            )
            continue

        profile = _profile_against_key(df, candidate)
        if profile.non_unique_rows == 0:
            unique_keys.append(candidate_set)
        profiles.append(profile)

    logger.info("=== KeyProfiler complete ===")
    return KeyProfileResult(
        raw_table_profile=raw_table_profile,
        key_profile=RankedCandidateProfiles(candidate_key_profiles=profiles),
    )
