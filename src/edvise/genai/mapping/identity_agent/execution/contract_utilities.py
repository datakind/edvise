"""
IdentityAgent execution helpers: apply grain and term contracts to DataFrames.

Composes :mod:`edvise.genai.mapping.identity_agent.grain_inference` (dedup, keys) with
:mod:`edvise.genai.mapping.identity_agent.term_normalization` (term ordering). For hooks, tests, and notebooks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import pandas as pd

from edvise.genai.mapping.identity_agent.grain_inference.deduplication import (
    drop_duplicate_keys,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermContract
from edvise.genai.mapping.identity_agent.term_normalization.term_order import (
    apply_term_order_from_config,
)
from edvise.utils.data_cleaning import convert_to_snake_case

logger = logging.getLogger(__name__)


def _map_key_after_student_id_rename(name: str, student_id_alias: str | None) -> str:
    """Align key names with ``student_id`` after :func:`~edvise.data_audit.custom_cleaning.clean_dataset` rename."""
    if not student_id_alias or not str(student_id_alias).strip():
        return name
    alias = str(student_id_alias).strip()
    alias_snake = convert_to_snake_case(alias)
    if name in ("student_id", alias, alias_snake):
        return "student_id"
    return name


def canonicalize_grain_contract_student_id_alias(
    contract: GrainContract,
) -> GrainContract:
    """
    Rewrite ``post_clean_primary_key``, ``join_keys_for_2a``, and ``dedup_policy.sort_by`` so
    any reference to ``contract.student_id_alias`` becomes ``student_id``.

    The dataframe passed to ``apply_grain_dedup`` already has ``student_id`` after cleaning;
    the grain contract may still name the pre-rename column in keys unless the model normalized them.
    """
    alias = contract.student_id_alias
    if not alias or not str(alias).strip():
        return contract

    pk = [
        _map_key_after_student_id_rename(k, alias)
        for k in contract.post_clean_primary_key
    ]
    jk = [_map_key_after_student_id_rename(k, alias) for k in contract.join_keys_for_2a]
    dp = contract.dedup_policy
    if dp.sort_by is None:
        new_sort: str | None = None
    else:
        new_sort = _map_key_after_student_id_rename(dp.sort_by, alias)
    new_dp = (
        dp if new_sort == dp.sort_by else dp.model_copy(update={"sort_by": new_sort})
    )

    if (
        pk == contract.post_clean_primary_key
        and jk == contract.join_keys_for_2a
        and new_dp is dp
    ):
        return contract

    return contract.model_copy(
        update={
            "post_clean_primary_key": pk,
            "join_keys_for_2a": jk,
            "dedup_policy": new_dp,
        }
    )


KeepArg = Literal["first", "last"]

# Grain contracts may name columns as the model guessed (e.g. ``TERM_DESC``) while
# ``clean_dataset`` has already applied ``normalize_columns`` (e.g. ``term_descr`` from
# ``TERM_DESCR``). Prefer exact / snake_case match; then a single safe prefix extension.
_MIN_KEY_LEN_FOR_PREFIX_FALLBACK = 8
_MAX_PREFIX_EXTENSION_CHARS = 6


def _resolve_grain_key_to_existing_column(name: str, columns: list[str]) -> str:
    """
    Map a grain-contract column name to a column present on the cleaned frame.

    Order: exact match → ``convert_to_snake_case`` match → unique prefix extension
    (same base name with a short suffix such as ``r`` in ``term_descr`` vs ``term_desc``).
    """
    colset = set(columns)
    if name in colset:
        return name
    normalized = convert_to_snake_case(name)
    if normalized in colset:
        return normalized
    if len(normalized) < _MIN_KEY_LEN_FOR_PREFIX_FALLBACK:
        raise ValueError(
            f"Grain key column {name!r} (as {normalized!r}) not in dataframe columns {sorted(colset)!r}"
        )
    candidates = [
        c
        for c in columns
        if c.startswith(normalized)
        and c != normalized
        and (len(c) - len(normalized)) <= _MAX_PREFIX_EXTENSION_CHARS
    ]
    if not candidates:
        raise ValueError(
            f"Grain key column {name!r} (as {normalized!r}) not in dataframe columns {sorted(colset)!r}"
        )
    candidates.sort(key=lambda c: (len(c) - len(normalized), c))
    best = candidates[0]
    best_delta = len(best) - len(normalized)
    if len(candidates) > 1:
        second_delta = len(candidates[1]) - len(normalized)
        if second_delta == best_delta:
            raise ValueError(
                f"Grain key {name!r} (as {normalized!r}) is ambiguous: multiple columns extend "
                f"that prefix: {candidates!r}"
            )
    if best != name and best != normalized:
        logger.info(
            "Resolved grain key %r → %r (prefix match on cleaned column names)",
            name,
            best,
        )
    return best


def _validate_key_columns(df: pd.DataFrame, keys: list[str], *, label: str) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns for grain key: {missing}")


def apply_grain_dedup(df: pd.DataFrame, contract: GrainContract) -> pd.DataFrame:
    """
    Apply ``contract.dedup_policy`` using ``post_clean_primary_key`` as the dedup key columns.

    - ``no_dedup``: return a copy unchanged.
    - ``policy_required``: collapse is needed but the rule is unresolved — do not deduplicate
      automatically; HITL must update the contract first (same no-op behavior as ``no_dedup`` here).
    - ``true_duplicate``: ``drop_duplicates`` on the key (optional ``sort_by`` for
      deterministic ordering before ``keep``).
    - ``temporal_collapse``: requires ``sort_by`` for meaningful ordering; if omitted, logs a
      warning and behaves like ``true_duplicate``.

    When ``dedupe_fn`` runs inside :func:`~edvise.data_audit.custom_cleaning.clean_dataset`,
    the frame already uses ``student_id``; :func:`canonicalize_grain_contract_student_id_alias`
    uses ``contract.student_id_alias`` (from IdentityAgent grain) to align key names with that column.
    """
    contract = canonicalize_grain_contract_student_id_alias(contract)
    policy = contract.dedup_policy
    cols = list(df.columns)
    try:
        keys = [
            _resolve_grain_key_to_existing_column(k, cols) for k in contract.unique_keys
        ]
    except ValueError as e:
        raise ValueError(f"apply_grain_dedup: {e}") from e
    _validate_key_columns(df, keys, label="apply_grain_dedup")

    if policy.strategy in ("no_dedup", "policy_required"):
        if policy.strategy == "policy_required":
            logger.warning(
                "dedup_policy.strategy=policy_required — skipping automatic dedup until HITL resolves policy"
            )
        return df.copy()

    sort_list: list[str] | None = None
    if policy.sort_by:
        try:
            sort_list = [_resolve_grain_key_to_existing_column(policy.sort_by, cols)]
        except ValueError as e:
            raise ValueError(f"apply_grain_dedup: dedup_policy.sort_by: {e}") from e
    if policy.strategy == "temporal_collapse" and not policy.sort_by:
        logger.warning(
            "temporal_collapse without dedup_policy.sort_by — using key-only dedup (keep=%s)",
            policy.keep or "first",
        )

    keep: KeepArg = policy.keep or "first"
    ascending = policy.sort_ascending if policy.sort_ascending is not None else True

    return drop_duplicate_keys(
        df,
        keys,
        keep=keep,
        sort_by=sort_list,
        ascending=ascending,
    )


def apply_term_order_from_contract(
    df: pd.DataFrame,
    term_pass: TermContract | None,
) -> pd.DataFrame:
    """
    Apply term-stage :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermContract` when
    ``term_config`` is set; otherwise return ``df`` unchanged.

    Delegates to :func:`~edvise.genai.mapping.identity_agent.term_normalization.term_order.apply_term_order_from_config`.
    Grain-stage :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.GrainContract` has no term fields.
    """
    if term_pass is None or term_pass.term_config is None:
        return df
    return apply_term_order_from_config(df, term_pass.term_config)


def apply_grain_execution(
    df: pd.DataFrame,
    grain: GrainContract,
    term_pass: TermContract | None = None,
) -> pd.DataFrame:
    """Run :func:`apply_grain_dedup` then :func:`apply_term_order_from_contract` (term contract, if any)."""
    out = apply_grain_dedup(df, grain)
    return apply_term_order_from_contract(out, term_pass)


def build_dedupe_fn_from_grain_contract(
    contract: GrainContract,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Build a ``dedupe_fn`` for :func:`~edvise.data_audit.custom_cleaning.clean_dataset` /
    :func:`~edvise.genai.mapping.schema_contract.build_from_school_config.build_schema_contract_from_config` that applies
    only the grain dedup step (not term order).

    Term order still runs later via ``term_order_fn`` / :func:`apply_term_order_from_contract` if needed.
    """

    def _dedupe_fn(frame: pd.DataFrame) -> pd.DataFrame:
        return apply_grain_dedup(frame, contract)

    return _dedupe_fn
