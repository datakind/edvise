"""
Optional row-level helpers for applying a ``DedupPolicy`` / cleaning pipelines.

Not required for Step 1 (profiling) or Step 2 (grain contract LLM). Schema-agnostic —
compose in pipelines or pass as ``custom_cleaning.CleanSpec.dedupe_fn``.
"""

from __future__ import annotations

import typing as t

import pandas as pd

KeepArg = t.Literal["first", "last"]


def drop_exact_row_duplicates(
    df: pd.DataFrame,
    *,
    keep: KeepArg | bool = "first",
    reset_index: bool = True,
) -> pd.DataFrame:
    """Drop fully identical rows (all columns)."""
    out = df.drop_duplicates(keep=keep)
    return out.reset_index(drop=True) if reset_index else out


def drop_duplicate_keys(
    df: pd.DataFrame,
    key_cols: list[str],
    *,
    keep: KeepArg = "first",
    sort_by: list[str] | None = None,
    ascending: bool | list[bool] = False,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Keep one row per distinct combination of ``key_cols``.

    When ``sort_by`` is set, rows are sorted (stable) before ``keep``, so
    ``keep="first"`` retains the preferred row within each key group.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"key_cols missing from DataFrame: {missing}")
    out = df.copy()
    if sort_by:
        miss_s = [c for c in sort_by if c not in out.columns]
        if miss_s:
            raise ValueError(f"sort_by missing from DataFrame: {miss_s}")
        out = out.sort_values(by=list(key_cols) + list(sort_by), ascending=ascending)
    out = out.drop_duplicates(subset=list(key_cols), keep=keep)
    return out.reset_index(drop=True) if reset_index else out


def suffix_disambiguate_within_keys(
    df: pd.DataFrame,
    key_cols: list[str],
    target_col: str,
    *,
    sep: str = "-",
    sort_within_group_by: list[str] | None = None,
    ascending: bool | list[bool] = True,
    only_rows_in_duplicate_key_groups: bool = True,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Append ``sep + 1-based index`` to ``target_col`` within each group sharing
    the same ``key_cols`` values (same pattern as PDP course renumbering).

    By default only rows that participate in a non-unique key (``duplicated``
    with ``keep=False``) are updated; singleton key rows are unchanged.

    For deterministic suffix order, pass ``sort_within_group_by`` (e.g. credits
    descending so the "primary" row becomes ``...-1``).

    **Uniqueness:** To obtain unique keys after this operation, ``target_col``
    should normally be one of ``key_cols`` (or downstream logic should extend
    the key to include it).
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"key_cols missing from DataFrame: {missing}")
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not in DataFrame")

    out = df.copy()
    dup_mask = out.duplicated(subset=list(key_cols), keep=False)
    if only_rows_in_duplicate_key_groups and not dup_mask.any():
        return out.reset_index(drop=True) if reset_index else out

    # Suffixes are string-concatenated (e.g. "37559-1"). Nullable Int64 and other
    # narrow dtypes reject that assignment; normalize before writing.
    out[target_col] = out[target_col].astype("string")

    if only_rows_in_duplicate_key_groups:
        work_idx = out.index[dup_mask]
        work = out.loc[work_idx]
    else:
        work = out

    if sort_within_group_by:
        miss = [c for c in sort_within_group_by if c not in work.columns]
        if miss:
            raise ValueError(f"sort_within_group_by missing from DataFrame: {miss}")
        sort_keys = list(key_cols) + list(sort_within_group_by)
        work = work.sort_values(by=sort_keys, ascending=ascending, kind="mergesort")

    grp_num = work.groupby(
        list(key_cols), sort=False, observed=True, dropna=False
    ).cumcount() + 1
    base = work[target_col].astype("string")
    new_vals = base.str.cat(grp_num.astype("string"), sep=sep)

    if only_rows_in_duplicate_key_groups:
        out.loc[work_idx, target_col] = new_vals
    else:
        out[target_col] = new_vals

    return out.reset_index(drop=True) if reset_index else out


def resolve_key_collisions(
    df: pd.DataFrame,
    key_cols: list[str],
    *,
    conflict_columns: list[str],
    disambiguate_column: str,
    drop_full_row_duplicates_first: bool = True,
    disambiguate_sep: str = "-",
    disambiguate_sort_by: list[str] | None = None,
    disambiguate_sort_ascending: bool | list[bool] = True,
    when_no_conflict_keep: KeepArg = "first",
    no_conflict_sort_by: list[str] | None = None,
    no_conflict_sort_ascending: bool = False,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Resolve duplicate *key* rows using two rules:

    1. If any ``conflict_columns`` value varies within a duplicate key group,
       keep all rows in that group and suffix ``disambiguate_column`` so keys
       can become unique (you should include ``disambiguate_column`` in
       ``key_cols``, or otherwise extend your key after this step).

    2. If duplicate keys agree on every ``conflict_column``, treat as redundant
       copies: keep one row per key (``when_no_conflict_keep``), optionally
       after sorting with ``no_conflict_sort_by`` (e.g. highest credits first).

    Optionally drops exact duplicate rows (all columns identical) first.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"key_cols missing from DataFrame: {missing}")
    if disambiguate_column not in df.columns:
        raise ValueError(
            f"disambiguate_column {disambiguate_column!r} not in DataFrame"
        )
    miss_c = [c for c in conflict_columns if c not in df.columns]
    if miss_c:
        raise ValueError(f"conflict_columns missing from DataFrame: {miss_c}")

    out = df.copy()
    if drop_full_row_duplicates_first:
        out = out.drop_duplicates(keep="first")

    dup_mask = out.duplicated(subset=list(key_cols), keep=False)
    if not dup_mask.any():
        return out.reset_index(drop=True) if reset_index else out

    if conflict_columns:
        nu = out.groupby(list(key_cols), observed=True, dropna=False)[
            list(conflict_columns)
        ].transform("nunique")
        grp_has_conflict = (nu > 1).any(axis=1)
    else:
        grp_has_conflict = pd.Series(False, index=out.index)

    conflict_row = dup_mask & grp_has_conflict
    simple_row = dup_mask & ~grp_has_conflict

    if conflict_row.any():
        conflict_idx = out.index[conflict_row]
        out[disambiguate_column] = out[disambiguate_column].astype("string")
        work = out.loc[conflict_idx].copy()
        if disambiguate_sort_by:
            miss = [c for c in disambiguate_sort_by if c not in work.columns]
            if miss:
                raise ValueError(
                    f"disambiguate_sort_by missing from DataFrame: {miss}"
                )
            sort_keys = list(key_cols) + list(disambiguate_sort_by)
            work = work.sort_values(
                by=sort_keys,
                ascending=disambiguate_sort_ascending,
                kind="mergesort",
            )
        grp_num = work.groupby(
            list(key_cols), sort=False, observed=True, dropna=False
        ).cumcount() + 1
        base = work[disambiguate_column].astype("string")
        out.loc[conflict_idx, disambiguate_column] = base.str.cat(
            grp_num.astype("string"), sep=disambiguate_sep
        )

    if simple_row.any():
        simple_idx = out.index[simple_row]
        simple_df = out.loc[simple_idx]
        if no_conflict_sort_by:
            miss = [c for c in no_conflict_sort_by if c not in simple_df.columns]
            if miss:
                raise ValueError(
                    f"no_conflict_sort_by missing from DataFrame: {miss}"
                )
            simple_df = simple_df.sort_values(
                by=list(key_cols) + list(no_conflict_sort_by),
                ascending=no_conflict_sort_ascending,
                kind="mergesort",
            )
        drop_ix = simple_df.index[simple_df.duplicated(list(key_cols), keep="first")]
        if len(drop_ix):
            out = out.drop(index=drop_ix)

    return out.reset_index(drop=True) if reset_index else out
