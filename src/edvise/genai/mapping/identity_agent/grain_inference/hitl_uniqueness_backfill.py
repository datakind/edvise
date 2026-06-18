"""
Repair ``hitl_context.candidate_keys[].uniqueness_score`` from :class:`RankedCandidateProfiles`.

Lives in its own module to avoid a circular import (``hitl.artifacts`` must not import
:mod:`.prompt`).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping

import pandas as pd

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainAmbiguityHITLContext,
    GrainCandidateKeyEntry,
    HITLDomain,
    HITLItem,
)
from edvise.genai.mapping.identity_agent.profiling import RankedCandidateProfiles
from edvise.genai.mapping.shared.profiling.grain_key_profile import (
    prepare_profiling_frame,
    profile_key,
)

logger = logging.getLogger(__name__)


def column_set_signature(columns: list[str]) -> tuple[str, ...]:
    """
    Key profile and HITL JSON use the same column names; the model can add leading/trailing
    space — normalize for matching the profiler.
    """
    return tuple(sorted(str(c).strip() for c in columns))


def _profile_uniqueness_by_column_sig(
    key_profile: RankedCandidateProfiles,
) -> dict[tuple[str, ...], float]:
    """Column-set signature (sorted, stripped) -> profiler ``uniqueness_score`` (last wins)."""
    m: dict[tuple[str, ...], float] = {}
    for prof in key_profile.candidate_key_profiles:
        sig = column_set_signature(prof.candidate_key.columns)
        m[sig] = prof.candidate_key.uniqueness_score
    return m


def _collect_candidate_key_signatures(items: list[HITLItem]) -> set[tuple[str, ...]]:
    sigs: set[tuple[str, ...]] = set()
    for item in items:
        if item.domain != HITLDomain.IDENTITY_GRAIN:
            continue
        ctx = item.hitl_context
        if not isinstance(ctx, GrainAmbiguityHITLContext):
            continue
        for entry in ctx.candidate_keys:
            sigs.add(column_set_signature(list(entry.columns)))
    return sigs


def _measured_uniqueness_by_signature(
    prepared: pd.DataFrame,
    signatures: set[tuple[str, ...]],
) -> dict[tuple[str, ...], float]:
    """Profile each column set once (sorted column order for stable compound keys)."""
    scores: dict[tuple[str, ...], float] = {}
    for sig in signatures:
        cols = list(sig)
        try:
            scores[sig] = profile_key(prepared, cols).uniqueness_score
        except ValueError as exc:
            logger.warning(
                "HITL uniqueness backfill (measured): skip columns=%s — %s",
                cols,
                exc,
            )
    return scores


def _apply_measured_scores_to_items(
    items: list[HITLItem],
    measured_by_sig: dict[tuple[str, ...], float],
) -> list[HITLItem]:
    if not measured_by_sig:
        return list(items)

    out: list[HITLItem] = []
    for item in items:
        if item.domain != HITLDomain.IDENTITY_GRAIN:
            out.append(item)
            continue
        ctx = item.hitl_context
        if not isinstance(ctx, GrainAmbiguityHITLContext):
            out.append(item)
            continue
        new_keys: list[GrainCandidateKeyEntry] = []
        for entry in ctx.candidate_keys:
            sig = column_set_signature(list(entry.columns))
            measured = measured_by_sig.get(sig)
            if measured is None:
                new_keys.append(entry)
                continue
            if entry.uniqueness_score != measured:
                logger.info(
                    "HITL uniqueness backfill (measured): item=%s rank=%d columns=%s %s -> %s",
                    item.item_id,
                    entry.rank,
                    entry.columns,
                    entry.uniqueness_score,
                    measured,
                )
            new_keys.append(entry.model_copy(update={"uniqueness_score": measured}))
        new_ctx = ctx.model_copy(update={"candidate_keys": new_keys})
        out.append(item.model_copy(update={"hitl_context": new_ctx}))
    return out


def backfill_hitl_uniqueness_scores_from_measured_keys(
    items: list[HITLItem],
    df: pd.DataFrame,
) -> list[HITLItem]:
    """
    Overwrite every ``hitl_context.candidate_keys[].uniqueness_score`` by profiling each
    distinct column set on the same deduped frame as KeyProfiler / verify.

    The LLM must list semantic grain alternatives for HITL, but should not invent uniqueness
    percentages — reviewers use those numbers to compare grains.
    """
    sigs = _collect_candidate_key_signatures(items)
    if not sigs:
        return list(items)
    prepared = prepare_profiling_frame(df)
    measured = _measured_uniqueness_by_signature(prepared, sigs)
    return _apply_measured_scores_to_items(items, measured)


def backfill_hitl_uniqueness_scores_by_table(
    items: list[HITLItem],
    dfs_by_table: Mapping[str, pd.DataFrame],
) -> list[HITLItem]:
    """Apply :func:`backfill_hitl_uniqueness_scores_from_measured_keys` per ``HITLItem.table``."""
    if not dfs_by_table:
        return list(items)

    by_table: dict[str, list[HITLItem]] = defaultdict(list)
    for it in items:
        by_table[str(it.table)].append(it)
    by_id: dict[str, HITLItem] = {}
    available_tables = sorted(dfs_by_table.keys())
    for table, group in by_table.items():
        df = dfs_by_table.get(table)
        if df is None:
            n_grain = sum(1 for it in group if it.domain == HITLDomain.IDENTITY_GRAIN)
            if n_grain:
                logger.warning(
                    "HITL uniqueness backfill (measured): no dataframe for table %r "
                    "(%d identity_grain item(s)); available: %s",
                    table,
                    n_grain,
                    available_tables,
                )
            for it in group:
                by_id[it.item_id] = it
        else:
            for it in backfill_hitl_uniqueness_scores_from_measured_keys(
                list(group), df
            ):
                by_id[it.item_id] = it
    return [by_id[it.item_id] for it in items]


def backfill_hitl_uniqueness_scores_from_key_profile(
    items: list[HITLItem],
    key_profile: RankedCandidateProfiles,
) -> list[HITLItem]:
    """
    Overwrite ``hitl_context.candidate_keys[].uniqueness_score`` when the column set matches
    a profiled :class:`~edvise.genai.mapping.identity_agent.profiling.schemas.CandidateKey`.

    The LLM often re-ranks keys for semantic grain and may emit **0** to mean "not the
    intended grain" or "requires collapse" — that misuses the field, which is **profiling
    evidence** (fraction of rows unique on the key), not a confidence score. The key profile
    is the source of truth for the same column list.
    """
    return _backfill_one_profile(items, key_profile)


def backfill_hitl_uniqueness_scores(
    items: list[HITLItem],
    key_profiles_by_table: Mapping[str, RankedCandidateProfiles] | None,
) -> list[HITLItem]:
    """
    Apply profiler uniqueness for each HITL item’s ``table`` using
    ``key_profiles_by_table[table]`` (e.g. ``{ \"student\": prof, \"course\": prof2 }``).

    Use :func:`backfill_hitl_uniqueness_scores_from_key_profile` when all items share one
    table/profile. If ``key_profiles_by_table`` is null or empty, return a shallow copy of
    ``items`` unchanged.
    """
    if not key_profiles_by_table:
        return list(items)

    by_table: dict[str, list[HITLItem]] = defaultdict(list)
    for it in items:
        by_table[str(it.table)].append(it)
    by_id: dict[str, HITLItem] = {}
    available_tables = sorted(key_profiles_by_table.keys())
    for table, group in by_table.items():
        profile = key_profiles_by_table.get(table)
        if profile is None:
            n_grain = sum(1 for it in group if it.domain == HITLDomain.IDENTITY_GRAIN)
            if n_grain:
                logger.warning(
                    "HITL uniqueness backfill: no key profile for table %r (%d identity_grain "
                    "item(s)); leaving candidate_keys uniqueness as-is. Profiling has tables: %s",
                    table,
                    n_grain,
                    available_tables,
                )
            for it in group:
                by_id[it.item_id] = it
        else:
            for it in _backfill_one_profile(list(group), profile):
                by_id[it.item_id] = it
    return [by_id[it.item_id] for it in items]


def _backfill_one_profile(
    items: list[HITLItem],
    key_profile: RankedCandidateProfiles,
) -> list[HITLItem]:
    profile_scores = _profile_uniqueness_by_column_sig(key_profile)
    if not profile_scores:
        return list(items)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "HITL uniqueness backfill: %d profiled key signatures for this table",
            len(profile_scores),
        )

    out: list[HITLItem] = []
    for item in items:
        if item.domain != HITLDomain.IDENTITY_GRAIN:
            out.append(item)
            continue
        ctx = item.hitl_context
        if not isinstance(ctx, GrainAmbiguityHITLContext):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "HITL item %s: skip backfill (hitl_context is not structured)",
                    item.item_id,
                )
            out.append(item)
            continue
        new_keys: list[GrainCandidateKeyEntry] = []
        for entry in ctx.candidate_keys:
            sig = column_set_signature(list(entry.columns))
            if sig in profile_scores:
                from_profile = profile_scores[sig]
                if entry.uniqueness_score != from_profile:
                    logger.info(
                        "HITL uniqueness backfill: item=%s rank=%d columns=%s %s -> %s",
                        item.item_id,
                        entry.rank,
                        entry.columns,
                        entry.uniqueness_score,
                        from_profile,
                    )
                new_keys.append(
                    entry.model_copy(update={"uniqueness_score": from_profile})
                )
            else:
                if entry.uniqueness_score == 0.0 and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "HITL uniqueness: no profile match for columns=%r (item=%s rank=%d)",
                        entry.columns,
                        item.item_id,
                        entry.rank,
                    )
                new_keys.append(entry)
        new_ctx = ctx.model_copy(update={"candidate_keys": new_keys})
        out.append(item.model_copy(update={"hitl_context": new_ctx}))
    return out
