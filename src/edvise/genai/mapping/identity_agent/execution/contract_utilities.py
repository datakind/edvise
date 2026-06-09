"""
IdentityAgent execution helpers: apply grain and term contracts to DataFrames.

Composes :mod:`edvise.genai.mapping.shared.grain.dedup_execution` (dedup, keys) with
:mod:`edvise.genai.mapping.identity_agent.term_normalization` (term ordering). For hooks, tests, and notebooks.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.shared.hitl.hook_spec.schemas import HookSpec
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermContract
from edvise.genai.mapping.identity_agent.term_normalization.term_order import (
    apply_term_order_from_config,
)
from edvise.genai.mapping.shared.grain.dedup_execution import (
    apply_categorical_priority,
    drop_duplicate_keys,
    suffix_repeat_course_identifier,
    validate_grain_key_columns,
)
from edvise.utils.data_cleaning import convert_to_snake_case

logger = logging.getLogger(__name__)


def _strip_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return None


def _map_key_after_canonical_learner_rename(
    name: str,
    learner_id_alias: str | None,
    *,
    canonical_column: str = "learner_id",
) -> str:
    """Align key names with the canonical learner column after :func:`~edvise.data_audit.custom_cleaning.clean_dataset` rename."""
    if canonical_column not in ("student_id", "learner_id"):
        raise ValueError(
            f"canonical_column must be 'student_id' or 'learner_id', got {canonical_column!r}"
        )
    name_snake = convert_to_snake_case(name)
    eff = _strip_optional_str(learner_id_alias)
    if eff is None:
        if canonical_column == "learner_id" and name_snake == "student_id":
            return "learner_id"
        return name
    alias_snake = convert_to_snake_case(eff)
    if canonical_column == "student_id":
        if name_snake in ("student_id", alias_snake):
            return "student_id"
        return name
    if name_snake in ("learner_id", "student_id", alias_snake):
        return "learner_id"
    return name


def canonicalize_grain_contract_learner_id_alias(
    contract: GrainContract,
    *,
    canonical_column: str = "learner_id",
    student_id_alias_fallback: str | None = None,
) -> GrainContract:
    """
    Rewrite ``post_clean_primary_key``, ``join_keys_for_2a``, and ``dedup_policy.sort_by``,
    ``dedup_policy.suffix_column``, and ``dedup_policy.priority_column`` so references to the
    learner identifier column match the
    canonical name after cleaning
    (default ``learner_id`` for GenAI; pass ``canonical_column`` as ``"student_id"`` for audit frames).

    The dataframe passed to ``apply_grain_dedup`` already uses that canonical column;
    the grain contract may still name the pre-rename column in keys unless the model normalized them.

    Grain ``learner_id_alias`` wins when set; ``student_id_alias_fallback`` supplies the same role
    when the contract omits it (e.g. use per-dataset ``DatasetConfig.student_id_alias``).
    Keys are matched with :func:`~edvise.utils.data_cleaning.convert_to_snake_case` so
    spellings such as ``STUDENT_ID`` align with ``student_id`` / the alias.
    """
    grain_alias = _strip_optional_str(contract.learner_id_alias)
    fb = _strip_optional_str(student_id_alias_fallback)
    effective_alias = grain_alias or fb

    pk = [
        _map_key_after_canonical_learner_rename(
            k, effective_alias, canonical_column=canonical_column
        )
        for k in contract.post_clean_primary_key
    ]
    jk = [
        _map_key_after_canonical_learner_rename(
            k, effective_alias, canonical_column=canonical_column
        )
        for k in contract.join_keys_for_2a
    ]
    dp = contract.dedup_policy
    if dp.sort_by is None:
        mapped_sort: str | None = None
    else:
        mapped_sort = _map_key_after_canonical_learner_rename(
            dp.sort_by, effective_alias, canonical_column=canonical_column
        )
    if dp.suffix_column is None:
        mapped_suffix: str | None = None
    else:
        mapped_suffix = _map_key_after_canonical_learner_rename(
            dp.suffix_column, effective_alias, canonical_column=canonical_column
        )
    if dp.priority_column is None:
        mapped_priority: str | None = None
    else:
        mapped_priority = _map_key_after_canonical_learner_rename(
            dp.priority_column, effective_alias, canonical_column=canonical_column
        )
    new_dp = (
        dp
        if (
            mapped_sort == dp.sort_by
            and mapped_suffix == dp.suffix_column
            and mapped_priority == dp.priority_column
        )
        else dp.model_copy(
            update={
                "sort_by": mapped_sort,
                "suffix_column": mapped_suffix,
                "priority_column": mapped_priority,
            }
        )
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


def _grain_dedup_function_names(functions: list[Any]) -> list[str]:
    names: list[str] = []
    for f in functions:
        if hasattr(f, "name"):
            names.append(f.name)
        else:
            names.append(str(f["name"]))
    return names


def _select_grain_dedup_function_name(functions: list[Any], table: str) -> str:
    names = _grain_dedup_function_names(functions)
    if not names:
        raise ValueError("hook_spec.functions is empty — cannot load grain dedup hook")
    if len(names) == 1:
        return names[0]
    t = table.lower().replace("-", "_")
    matches = [n for n in names if t in n.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"hook_spec.functions has {len(names)} entries {names!r}; expected exactly one, or "
        f"exactly one whose name contains table {table!r}."
    )


def load_grain_dedup_hook_from_hook_spec(
    hook_spec: HookSpec | dict[str, Any],
    *,
    modules_root: str | Path,
    table: str,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Import the materialized grain dedup module and return the single dedup callable.

    The hook is invoked once per key-group: ``df.groupby(keys).apply`` passes each group
    ``DataFrame`` (same columns as the full frame) to the loaded function, which must return
    a ``DataFrame`` (typically zero or one row per group) with the **same columns** as the
    input group (Option B: do not drop columns in custom dedup). Column names are **already
    snake_case** — :func:`~edvise.data_audit.custom_cleaning.clean_dataset` runs
    ``normalize_columns`` before ``dedupe_fn``.
    """
    from edvise.genai.mapping.shared.hitl.hook_spec.paths import (
        resolve_hook_module_path,
    )

    hs = (
        hook_spec.model_dump(mode="json")
        if isinstance(hook_spec, HookSpec)
        else dict(hook_spec)
    )
    rel = hs.get("file")
    if not rel:
        raise ValueError(
            "hook_spec.file is required to load grain dedup hook from disk"
        )
    funcs = hs.get("functions") or []
    fn_name = _select_grain_dedup_function_name(funcs, table)
    path = resolve_hook_module_path(rel, root=modules_root)
    spec = importlib.util.spec_from_file_location("_ia_grain_dedup_hooks", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load hook module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise ValueError(f"Module {path} missing callable {fn_name!r}")
    return cast(Callable[[pd.DataFrame], pd.DataFrame], fn)


def _apply_grain_hook_dedup_by_key_groups(
    df: pd.DataFrame,
    keys: list[str],
    hook_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """``df`` uses normalized column names (see :func:`apply_grain_dedup`)."""
    if df.empty:
        return df.copy()
    pieces: list[pd.DataFrame] = []
    for _, group in df.groupby(keys, dropna=False, sort=False):
        out_g = hook_fn(group)
        if not isinstance(out_g, pd.DataFrame):
            raise TypeError(
                "grain dedup hook must return pandas.DataFrame, "
                f"got {type(out_g).__name__}"
            )
        pieces.append(out_g)
    return pd.concat(pieces, ignore_index=True)


def apply_grain_dedup(
    df: pd.DataFrame,
    contract: GrainContract,
    *,
    canonical_learner_column: str = "learner_id",
    student_id_alias_fallback: str | None = None,
    hook_modules_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply ``contract.dedup_policy`` using ``post_clean_primary_key`` as the dedup key columns.

    - ``no_dedup``: return a copy unchanged.
    - ``policy_required`` **without** ``dedup_policy.hook_spec``: do not deduplicate automatically
      (HITL still resolving policy).
    - ``policy_required`` **with** ``dedup_policy.hook_spec``: load ``hook_spec.file`` under
      ``hook_modules_root`` (e.g. ``SchoolMappingConfig.bronze_volumes_path``), import the named
      dedup function, and run it **once per key-group** (``groupby`` on ``post_clean_primary_key``).
      The callable receives each group's ``DataFrame`` and must return a ``DataFrame``.
    - ``true_duplicate``: ``drop_duplicates`` on the key; contract must not set ``sort_by`` or
      ``keep`` (defaults to first row per key in key order / pandas semantics).
    - ``temporal_collapse``: time / sequence-like ``sort_by``; sort then dedupe on the key (keep first).
    - ``first_by_column``: non-time ``sort_by`` (e.g. grades); same mechanics as temporal_collapse but
      distinct strategy for contracts and analytics.
    - ``categorical_priority``: one row per key, chosen by ranking ``dedup_policy.priority_column``
      values with ``dedup_policy.priority_order`` (highest-priority first in the list; exact
      match, else substring: longest token wins, then first-listed).
    - ``suffix_identifier``: no rows dropped. Within each key group, ``dedup_policy.suffix_column``
      values are suffixed with ``-1``, ``-2``, ... in input row order so the identifier is unique
      (e.g. repeat course enrollments).

    When ``dedupe_fn`` runs inside :func:`~edvise.data_audit.custom_cleaning.clean_dataset`,
    the frame already uses the canonical learner column (default ``learner_id`` for GenAI, or
    ``student_id`` when ``canonical_learner_column`` is set accordingly); :func:`canonicalize_grain_contract_learner_id_alias`
    uses ``contract.learner_id_alias`` (grain wins when set) and optional ``student_id_alias_fallback``
    (e.g. per-dataset ``DatasetConfig.student_id_alias``) when the contract omits it; key tokens are
    snake-cased before matching alias / canonical names.
    """
    contract = canonicalize_grain_contract_learner_id_alias(
        contract,
        canonical_column=canonical_learner_column,
        student_id_alias_fallback=student_id_alias_fallback,
    )
    policy = contract.dedup_policy
    cols = list(df.columns)
    try:
        keys = [
            _resolve_grain_key_to_existing_column(k, cols) for k in contract.unique_keys
        ]
    except ValueError as e:
        raise ValueError(f"apply_grain_dedup: {e}") from e
    validate_grain_key_columns(df, keys, label="apply_grain_dedup")

    if policy.strategy == "no_dedup":
        return df.copy()

    if policy.strategy == "policy_required":
        if policy.hook_spec is not None:
            if hook_modules_root is None:
                raise ValueError(
                    "dedup_policy has hook_spec but hook_modules_root was not passed. "
                    "Pass hook_modules_root= (e.g. SchoolMappingConfig.bronze_volumes_path) "
                    "so identity_hooks/.../dedup_hooks.py can be imported."
                )
            hook_fn = load_grain_dedup_hook_from_hook_spec(
                policy.hook_spec,
                modules_root=hook_modules_root,
                table=contract.table,
            )
            return _apply_grain_hook_dedup_by_key_groups(df, keys, hook_fn)
        logger.warning(
            "dedup_policy.strategy=policy_required — skipping automatic dedup until HITL resolves policy"
        )
        return df.copy()

    if policy.strategy == "categorical_priority":
        pc = policy.priority_column
        po = policy.priority_order
        if not pc or not str(pc).strip() or not po or not len(po):
            raise ValueError(
                "apply_grain_dedup: categorical_priority requires non-null "
                "dedup_policy.priority_column and non-empty priority_order (schema should enforce this)"
            )
        try:
            priority_resolved = _resolve_grain_key_to_existing_column(pc, cols)
        except ValueError as e:
            raise ValueError(
                f"apply_grain_dedup: dedup_policy.priority_column: {e}"
            ) from e
        return apply_categorical_priority(
            df,
            group_by=keys,
            priority_column=priority_resolved,
            priority_order=po,
        )

    if policy.strategy == "suffix_identifier":
        sc = policy.suffix_column
        if not sc or not str(sc).strip():
            raise ValueError(
                "apply_grain_dedup: suffix_identifier requires dedup_policy.suffix_column"
            )
        try:
            suffix_resolved = _resolve_grain_key_to_existing_column(sc, cols)
        except ValueError as e:
            raise ValueError(
                f"apply_grain_dedup: dedup_policy.suffix_column: {e}"
            ) from e
        return suffix_repeat_course_identifier(
            df, group_by=keys, suffix_column=suffix_resolved
        )

    sort_list: list[str] | None = None
    if policy.sort_by:
        try:
            sort_list = [_resolve_grain_key_to_existing_column(policy.sort_by, cols)]
        except ValueError as e:
            raise ValueError(f"apply_grain_dedup: dedup_policy.sort_by: {e}") from e
    if (
        policy.strategy in ("temporal_collapse", "first_by_column")
        and not policy.sort_by
    ):
        logger.warning(
            "%s without dedup_policy.sort_by — using key-only dedup (keep=%s)",
            policy.strategy,
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
    *,
    hook_modules_root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Apply term-stage :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermContract` when
    ``term_config`` is set; otherwise return ``df`` unchanged.

    Runs primary ``term_config`` (entry / cohort term) when present.

    Delegates to :func:`~edvise.genai.mapping.identity_agent.term_normalization.term_order.apply_term_order_from_config`.
    Grain-stage :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.GrainContract` has no term fields.
    """
    if term_pass is None:
        return df
    out = df
    if term_pass.term_config is not None:
        out = apply_term_order_from_config(
            out, term_pass.term_config, hook_modules_root=hook_modules_root
        )
    return out


def apply_grain_execution(
    df: pd.DataFrame,
    grain: GrainContract,
    term_pass: TermContract | None = None,
    *,
    canonical_learner_column: str = "learner_id",
    student_id_alias_fallback: str | None = None,
    hook_modules_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run :func:`apply_grain_dedup` then :func:`apply_term_order_from_contract` (term contract, if any)."""
    out = apply_grain_dedup(
        df,
        grain,
        canonical_learner_column=canonical_learner_column,
        student_id_alias_fallback=student_id_alias_fallback,
        hook_modules_root=hook_modules_root,
    )
    return apply_term_order_from_contract(
        out, term_pass, hook_modules_root=hook_modules_root
    )


def build_dedupe_fn_from_grain_contract(
    contract: GrainContract,
    *,
    canonical_learner_column: str = "learner_id",
    student_id_alias_fallback: str | None = None,
    hook_modules_root: str | Path | None = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Build a ``dedupe_fn`` for :func:`~edvise.data_audit.custom_cleaning.clean_dataset` /
    :func:`~edvise.genai.mapping.shared.schema_contract.build_from_school_config.build_schema_contract_from_config` that applies
    only the grain dedup step (not term order).

    Term order still runs later via ``term_order_fn`` / :func:`apply_term_order_from_contract` if needed.
    Pass ``canonical_learner_column=\"student_id\"`` only when the cleaned frame still uses
    :func:`~edvise.data_audit.custom_cleaning.clean_dataset` audit defaults (``student_id``).

    Pass ``hook_modules_root`` when ``dedup_policy.hook_spec`` is set (materialized ``dedup_hooks.py``).

    ``student_id_alias_fallback`` forwards to :func:`apply_grain_dedup` when the grain contract
    omits ``learner_id_alias``.
    """

    def _dedupe_fn(frame: pd.DataFrame) -> pd.DataFrame:
        return apply_grain_dedup(
            frame,
            contract,
            canonical_learner_column=canonical_learner_column,
            student_id_alias_fallback=student_id_alias_fallback,
            hook_modules_root=hook_modules_root,
        )

    return _dedupe_fn
