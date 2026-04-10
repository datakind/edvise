"""Grain-stage deduplication on primary/join keys (used by ``apply_grain_dedup``)."""

from __future__ import annotations

from typing import Literal

import pandas as pd


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
