"""Row-level helpers for grain deduplication and key collision handling."""

from __future__ import annotations

from typing import Literal

import pandas as pd


def drop_exact_row_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are identical across all columns."""
    return df.drop_duplicates().reset_index(drop=True)


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
    return out.reset_index(drop=True)


def suffix_disambiguate_within_keys(
    df: pd.DataFrame,
    key_cols: list[str],
    target_col: str,
    *,
    sort_within_group_by: list[str] | None = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Within each group sharing ``key_cols``, append ``-1``, ``-2``, … to ``target_col``
    when the group has more than one row. Single-row groups are unchanged.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"key_cols missing from frame: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} missing from frame")

    out = df.copy()
    # Nullable Int64 (and other integer dtypes) cannot hold suffixed strings — promote first.
    if pd.api.types.is_integer_dtype(out[target_col]):
        out[target_col] = out[target_col].astype("string")

    grouped = out.groupby(key_cols, dropna=False, sort=False)

    for _, g in grouped:
        if len(g) <= 1:
            continue
        g_ord = (
            g.sort_values(sort_within_group_by, ascending=ascending)
            if sort_within_group_by
            else g
        )
        for rank, idx in enumerate(g_ord.index, start=1):
            raw = out.at[idx, target_col]
            base = "" if pd.isna(raw) else str(raw)
            out.at[idx, target_col] = f"{base}-{rank}"

    return out


def resolve_key_collisions(
    df: pd.DataFrame,
    key_cols: list[str],
    conflict_columns: list[str],
    disambiguate_column: str,
) -> pd.DataFrame:
    """
    Rows that share ``key_cols`` and identical ``conflict_columns`` are collapsed to one row.
    If ``conflict_columns`` differ within a key group, ``disambiguate_column`` is suffixed
    (via :func:`suffix_disambiguate_within_keys`) so the composite key becomes unique.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"key_cols missing from frame: {missing}")
    for c in conflict_columns:
        if c not in df.columns:
            raise ValueError(f"conflict column {c!r} missing from frame")
    if disambiguate_column not in df.columns:
        raise ValueError(
            f"disambiguate_column {disambiguate_column!r} missing from frame"
        )

    parts: list[pd.DataFrame] = []
    for _, g in df.groupby(key_cols, dropna=False, sort=False):
        if len(g) == 1:
            parts.append(g)
            continue
        distinct_conflicts = g[conflict_columns].drop_duplicates()
        if len(distinct_conflicts) == 1:
            parts.append(g.iloc[[0]])
        else:
            parts.append(
                suffix_disambiguate_within_keys(
                    g.reset_index(drop=True),
                    key_cols,
                    disambiguate_column,
                )
            )
    return pd.concat(parts, ignore_index=True)
