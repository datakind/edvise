"""
Shared grain dedup **execution** (pandas row operations).

Used by IdentityAgent :func:`~edvise.genai.mapping.identity_agent.execution.contract_utilities.apply_grain_dedup`
and SMA :func:`apply_sma_grain_resolution_payload` so collapse / suffix / categorical semantics stay aligned.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import pandas as pd

logger = logging.getLogger(__name__)

KeepArg = Literal["first", "last"]


def drop_duplicate_keys(
    df: pd.DataFrame,
    key_cols: list[str],
    *,
    keep: Literal["first", "last"] = "first",
    sort_by: list[str] | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Drop duplicate rows sharing the same ``key_cols``, optionally sorting first so
    ``keep`` is applied in a deterministic semantic order (e.g. highest score wins).

    **Option B (grain dedup):** reduces **row count** only; output columns match input.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"key_cols missing from frame: {missing}")
    if sort_by:
        bad = [c for c in sort_by if c not in df.columns]
        if bad:
            raise ValueError(f"sort_by columns missing from frame: {bad}")
    work = df.copy()
    if sort_by:
        work = work.sort_values(by=sort_by, ascending=ascending)
    out = work.drop_duplicates(subset=key_cols, keep=keep)
    if not out.columns.equals(df.columns):
        raise AssertionError(
            "drop_duplicate_keys must preserve all input columns (temporal_collapse / grain dedup "
            f"does not delete columns); got {list(out.columns)} vs {list(df.columns)}"
        )
    return out.reset_index(drop=True)


def validate_grain_key_columns(
    df: pd.DataFrame, keys: list[str], *, label: str
) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns for grain key: {missing}")


def assert_suffix_column_in_entity_keys(
    suffix_column: str | None, entity_keys: list[str]
) -> str:
    """
    ``suffix_identifier`` may only suffix a column that is already part of the manifest
    entity grain (execution ``entity_keys`` / SMA ``manifest_source_keys``).

    Raises:
        ValueError: If ``suffix_column`` is missing or not in ``entity_keys``.
    """
    if not suffix_column or not str(suffix_column).strip():
        raise ValueError("suffix_identifier requires a non-empty suffix_column.")
    sc = str(suffix_column).strip()
    if sc not in entity_keys:
        raise ValueError(
            f"suffix_column {sc!r} must be one of the manifest entity grain columns {entity_keys!r}"
        )
    return sc


def suffix_repeat_course_identifier(
    df: pd.DataFrame,
    group_by: list[str],
    suffix_column: str,
) -> pd.DataFrame:
    """
    Within each ``group_by`` key group, append ``-1``, ``-2``, ... to ``suffix_column`` in
    dataframe row order (no sorting) when the group has more than one row. Single-row groups
    and all other columns are unchanged. Row count is preserved.
    """
    validate_grain_key_columns(df, group_by, label="suffix_repeat_course_identifier")
    if suffix_column not in df.columns:
        raise ValueError(
            f"suffix_repeat_course_identifier: suffix_column {suffix_column!r} "
            f"not in dataframe columns{sorted(df.columns)!r}"
        )
    out = df.copy()
    if out.empty or len(out) < 2:
        return out
    out[suffix_column] = out[suffix_column].astype("string")
    grp = out.groupby(group_by, dropna=False, sort=False)
    n_in_group = grp[suffix_column].transform("size")
    rank_1based = grp.cumcount() + 1
    should_suffix = n_in_group > 1
    if not should_suffix.any():
        return out
    new_vals = [
        f"{base}-{r}"
        for base, r in zip(
            out.loc[should_suffix, suffix_column].tolist(),
            rank_1based[should_suffix].tolist(),
        )
    ]
    out.loc[should_suffix, suffix_column] = new_vals
    return out


def _dedup_categorical_values_equal(a: object, b: object) -> bool:
    if a is b:
        return True
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _categorical_value_rank(value: object, priority_order: list[str]) -> int:
    for i, p in enumerate(priority_order):
        if _dedup_categorical_values_equal(value, p):
            return i
    if not isinstance(value, str):
        return len(priority_order)
    best: tuple[int, int] | None = None
    for i, p in enumerate(priority_order):
        if not isinstance(p, str) or not p:
            continue
        if p in value:
            key = (-len(p), i)
            if best is None or key < best:
                best = key
    if best is not None:
        return best[1]
    return len(priority_order)


def apply_categorical_priority(
    df: pd.DataFrame,
    group_by: list[str],
    priority_column: str,
    priority_order: list[str],
) -> pd.DataFrame:
    """
    Within each ``group_by`` key group, keep a single row: the one with the best (lowest)
    rank on ``priority_column``, where ``priority_order`` lists values from highest to
    lowest priority. Exact match first, else substring (longest token wins, then first-listed).
    """
    validate_grain_key_columns(df, group_by, label="apply_categorical_priority")
    if priority_column not in df.columns:
        raise ValueError(
            f"apply_categorical_priority: priority_column {priority_column!r} not in "
            f"columns {sorted(df.columns)!r}"
        )
    if not priority_order or not len(priority_order):
        raise ValueError("apply_categorical_priority: priority_order must be non-empty")
    if df.empty:
        return df.copy()
    work = df.copy()
    work["_categorical_dedup_rank_"] = work[priority_column].map(
        lambda v: _categorical_value_rank(v, priority_order)
    )
    sort_keys = list(group_by) + ["_categorical_dedup_rank_"]
    work = work.sort_values(by=sort_keys, kind="mergesort")
    out = work.drop_duplicates(subset=group_by, keep="first").drop(
        columns=["_categorical_dedup_rank_"]
    )
    return out.reset_index(drop=True)


def _sma_grain_resolution_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_list = payload.get("grain_resolutions")
    if isinstance(raw_list, list) and raw_list:
        return [cast(dict[str, Any], dict(x)) for x in raw_list if isinstance(x, dict)]
    single = payload.get("grain_resolution")
    if isinstance(single, dict):
        return [cast(dict[str, Any], dict(single))]
    if isinstance(payload, dict) and payload.get("dedup_strategy") is not None:
        return [cast(dict[str, Any], dict(payload))]
    return []


def _apply_single_sma_grain_resolution(
    base_df: pd.DataFrame,
    entity_keys: list[str],
    gr: dict[str, Any],
    *,
    log: logging.Logger,
    step_index: int,
) -> pd.DataFrame:
    """Apply one ``grain_resolution`` dict to ``base_df`` (SMA onboard sidecar step)."""
    strategy = gr.get("dedup_strategy")
    if strategy is None:
        return base_df.copy()

    if not entity_keys:
        log.warning(
            "sma_grain_resolution[%s]: empty entity_keys — skipping reduction",
            step_index,
        )
        return base_df.copy()

    missing = [k for k in entity_keys if k not in base_df.columns]
    if missing:
        log.warning(
            "sma_grain_resolution[%s]: entity_keys not in base_df %s — skipping reduction",
            step_index,
            missing,
        )
        return base_df.copy()

    if strategy == "no_dedup":
        return base_df.copy()

    if strategy == "suffix_identifier":
        suffix_col = gr.get("suffix_column")
        try:
            sc = assert_suffix_column_in_entity_keys(suffix_col, entity_keys)
        except ValueError as e:
            log.warning(
                "sma_grain_resolution[%s]: suffix_identifier invalid — %s",
                step_index,
                e,
            )
            return base_df.copy()
        if sc not in base_df.columns:
            log.warning(
                "sma_grain_resolution[%s]: suffix_column %r not in base_df — skipping",
                step_index,
                sc,
            )
            return base_df.copy()
        try:
            return suffix_repeat_course_identifier(base_df, entity_keys, sc)
        except ValueError as e:
            log.warning(
                "sma_grain_resolution[%s]: suffix_identifier failed: %s", step_index, e
            )
            return base_df.copy()

    if strategy == "intentional_step_down":
        return drop_duplicate_keys(base_df, entity_keys, keep="first")

    if strategy == "true_duplicate":
        return drop_duplicate_keys(base_df, entity_keys, keep="first")

    if strategy in ("temporal_collapse", "first_by_column"):
        sort_by = gr.get("dedup_sort_by")
        asc = gr.get("dedup_sort_ascending")
        if not sort_by or not str(sort_by).strip() or asc is None:
            log.warning(
                "sma_grain_resolution[%s]: %s missing dedup_sort_by or dedup_sort_ascending — "
                "no reduction",
                step_index,
                strategy,
            )
            return base_df.copy()
        col = str(sort_by).strip()
        if col not in base_df.columns:
            log.warning(
                "sma_grain_resolution[%s]: dedup_sort_by %r not in base_df — no reduction",
                step_index,
                col,
            )
            return base_df.copy()
        keep_raw = gr.get("dedup_keep")
        keep: KeepArg = "first"
        if keep_raw in ("first", "last"):
            keep = cast(Literal["first", "last"], keep_raw)
        elif keep_raw is not None:
            log.warning(
                "sma_grain_resolution[%s]: invalid dedup_keep %r — using 'first'",
                step_index,
                keep_raw,
            )
        return drop_duplicate_keys(
            base_df,
            entity_keys,
            keep=keep,
            sort_by=[col],
            ascending=bool(asc),
        )

    if strategy == "categorical_priority":
        pc = gr.get("priority_column")
        po = gr.get("priority_order")
        if not pc or not str(pc).strip():
            log.warning(
                "sma_grain_resolution[%s]: categorical_priority missing priority_column — skipping",
                step_index,
            )
            return base_df.copy()
        if not isinstance(po, list) or len(po) == 0:
            log.warning(
                "sma_grain_resolution[%s]: categorical_priority missing priority_order — skipping",
                step_index,
            )
            return base_df.copy()
        priority_col = str(pc).strip()
        order = [str(x) for x in po]
        if priority_col not in base_df.columns:
            log.warning(
                "sma_grain_resolution[%s]: priority_column %r not in base_df — skipping",
                step_index,
                priority_col,
            )
            return base_df.copy()
        try:
            return apply_categorical_priority(base_df, entity_keys, priority_col, order)
        except ValueError as e:
            log.warning(
                "sma_grain_resolution[%s]: categorical_priority failed: %s",
                step_index,
                e,
            )
            return base_df.copy()

    log.warning(
        "sma_grain_resolution[%s]: unknown dedup_strategy %r — ignoring",
        step_index,
        strategy,
    )
    return base_df.copy()


def apply_sma_grain_resolution_payload(
    base_df: pd.DataFrame,
    entity_keys: list[str],
    payload: dict[str, Any],
    *,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Apply SMA grain resolution step(s) from an ``sma_grain_resolution_*.json`` sidecar dict.

    If ``grain_resolutions`` (list of objects) is present, steps are applied **in order**.
    Otherwise a single ``grain_resolution`` object is applied (legacy sidecar shape).
    """
    log = log or logger
    steps = _sma_grain_resolution_steps(payload)
    if not steps:
        log.warning("sma_grain_resolution: no grain_resolution steps — ignoring")
        return base_df.copy()

    if not entity_keys:
        log.warning("sma_grain_resolution: empty entity_keys — skipping reduction")
        return base_df.copy()

    missing = [k for k in entity_keys if k not in base_df.columns]
    if missing:
        log.warning(
            "sma_grain_resolution: entity_keys not in base_df %s — skipping reduction",
            missing,
        )
        return base_df.copy()

    work = base_df.copy()
    for i, gr in enumerate(steps):
        work = _apply_single_sma_grain_resolution(
            work, entity_keys, gr, log=log, step_index=i
        )
    return work


__all__ = [
    "apply_categorical_priority",
    "apply_sma_grain_resolution_payload",
    "assert_suffix_column_in_entity_keys",
    "drop_duplicate_keys",
    "suffix_repeat_course_identifier",
    "validate_grain_key_columns",
]
