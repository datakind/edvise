"""In-memory grain dedup and term-order transforms (for hooks, tests, or notebooks)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import pandas as pd

from edvise.feature_generation.term import add_term_order
from edvise.genai.identity_agent.grain_contract.deduplication import drop_duplicate_keys
from edvise.genai.identity_agent.grain_contract.schemas import DedupStrategy, IdentityGrainContract

logger = logging.getLogger(__name__)

KeepArg = Literal["first", "last"]


def _validate_key_columns(df: pd.DataFrame, keys: list[str], *, label: str) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns for grain key: {missing}")


def apply_grain_dedup(df: pd.DataFrame, contract: IdentityGrainContract) -> pd.DataFrame:
    """
    Apply ``contract.dedup_policy`` using ``post_clean_primary_key`` as the dedup key columns.

    - ``no_dedup``: return a copy unchanged.
    - ``true_duplicate``: ``drop_duplicates`` on the key (optional ``sort_by`` for
      deterministic ordering before ``keep``).
    - ``temporal_collapse``: requires ``sort_by`` for meaningful ordering; if omitted, logs a
      warning and behaves like ``true_duplicate``.
    """
    policy = contract.dedup_policy
    keys = list(contract.unique_keys)
    _validate_key_columns(df, keys, label="apply_grain_dedup")

    if policy.strategy == DedupStrategy.no_dedup:
        return df.copy()

    sort_list: list[str] | None = [policy.sort_by] if policy.sort_by else None
    if policy.strategy == DedupStrategy.temporal_collapse and not policy.sort_by:
        logger.warning(
            "temporal_collapse without dedup_policy.sort_by — using key-only dedup (keep=%s)",
            policy.keep or "first",
        )

    keep: KeepArg = (policy.keep or "first") if policy.keep in ("first", "last") else "first"

    return drop_duplicate_keys(
        df,
        keys,
        keep=keep,
        sort_by=sort_list,
        ascending=True,
    )


def apply_grain_term_order(df: pd.DataFrame, contract: IdentityGrainContract) -> pd.DataFrame:
    """
    If ``contract.term_order_column`` is set, run :func:`~edvise.feature_generation.term.add_term_order`.

    Adds ``season``, ``year``, ``season_order``, ``is_core_term``, ``term_order`` (see that
    function). If ``term_order_column`` is ``None``, returns ``df`` unchanged.
    """
    col = contract.term_order_column
    if not col:
        return df
    if col not in df.columns:
        raise ValueError(f"apply_grain_term_order: column {col!r} not in DataFrame")
    return add_term_order(df, term_col=col)


def apply_grain_execution(df: pd.DataFrame, contract: IdentityGrainContract) -> pd.DataFrame:
    """Run :func:`apply_grain_dedup` then :func:`apply_grain_term_order`."""
    out = apply_grain_dedup(df, contract)
    return apply_grain_term_order(out, contract)


def build_dedupe_fn_from_grain_contract(
    contract: IdentityGrainContract,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Build a ``dedupe_fn`` for :func:`~edvise.data_audit.custom_cleaning.clean_dataset` / SMA
    preprocessing that applies only the grain dedup step (not term order).

    Term order still runs later via ``term_order_fn`` / :func:`apply_grain_term_order` if needed.
    """

    def _dedupe_fn(frame: pd.DataFrame) -> pd.DataFrame:
        return apply_grain_dedup(frame, contract)

    return _dedupe_fn
