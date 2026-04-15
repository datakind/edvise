"""
Grain merge, frozen schema-contract build, and enriched training payloads for GenAI.

Combines :func:`merge_grain_contracts_into_school_config` / :func:`build_schema_contract_from_grain_contracts`
with helpers that attach historical-example metadata (samples, null stats, low-cardinality uniques)
for Schema Mapping Agent prompts.

The **canonical JSON shape** consumed by Schema Mapping Agent is
:class:`~edvise.genai.mapping.schema_contract.EnrichedSchemaContractForSMA`
(enriched institution file with ``school_id`` and per-dataset ``training``).

Use :func:`build_enriched_schema_contract_for_institution` for one JSON per institution
(all logical datasets under ``datasets``), or :func:`build_enriched_schema_contract_for_dataset`
for a single-dataset slice. Then :func:`save_enriched_schema_contract`.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from edvise.configs.custom import CleaningConfig
from edvise.configs.genai import (
    DatasetConfig,
    SchoolMappingConfig,
    resolve_genai_data_path,
)
from edvise.data_audit.custom_cleaning import DtypeGenerationOptions, TermOrderFn
from edvise.genai.mapping.identity_agent.execution.contract_utilities import (
    build_dedupe_fn_from_grain_contract,
    canonicalize_grain_contract_learner_id_alias,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.utils.data_cleaning import convert_to_snake_case

logger = logging.getLogger(__name__)


def merge_grain_learner_id_alias_into_school_config(
    school_config: SchoolMappingConfig,
    grain_contracts_by_dataset: dict[str, GrainContract],
) -> SchoolMappingConfig:
    """
    Set ``school_config.cleaning.student_id_alias`` from grain ``learner_id_alias`` when specified.

    Grain JSON uses ``learner_id_alias``; that value is written to ``CleaningConfig.student_id_alias``
    for :func:`~edvise.data_audit.custom_cleaning.clean_dataset`, which maps the alias to the
    canonical person column (``learner_id`` when using GenAI defaults).

    All non-null ``learner_id_alias`` values across the given contracts must agree; otherwise
    raises. When ``inputs.toml`` already sets a different alias, grain wins and a warning is logged.

    Callers that build schema contracts **one dataset at a time** (e.g. manifest helpers) should
    run this once per institution with the **full** grain map so ``cleaning`` matches every table,
    then pass a single-dataset slice into :func:`merge_grain_contracts_into_school_config` /
    :func:`build_schema_contract_from_grain_contracts` for primary-key overrides.
    """
    aliases: set[str] = set()
    for gc in grain_contracts_by_dataset.values():
        if gc.learner_id_alias:
            aliases.add(gc.learner_id_alias.strip())
    if not aliases:
        return school_config
    if len(aliases) > 1:
        raise ValueError(
            "Grain contracts disagree on learner_id_alias: "
            f"{sorted(aliases)}. Resolve before merging into school config."
        )
    alias = next(iter(aliases))
    existing = (
        school_config.cleaning.student_id_alias if school_config.cleaning else None
    )
    if existing and existing != alias:
        logger.warning(
            "Overriding school_config.cleaning.student_id_alias %r with grain contract value %r",
            existing,
            alias,
        )
    base = school_config.cleaning or CleaningConfig()
    new_cleaning = base.model_copy(update={"student_id_alias": alias})
    return school_config.model_copy(update={"cleaning": new_cleaning})


def merge_grain_contracts_into_school_config(
    school_config: SchoolMappingConfig,
    grain_contracts_by_dataset: dict[str, GrainContract],
    *,
    dataset_name_suffix: str = "",
    canonical_learner_column: Literal["student_id", "learner_id"] = "learner_id",
) -> SchoolMappingConfig:
    """
    Return a copy of ``school_config`` with ``primary_keys`` overridden from grain contracts,
    and ``cleaning.student_id_alias`` set when grain contracts provide a consistent alias.

    Only datasets **present** in ``grain_contracts_by_dataset`` are updated; others keep
    ``inputs.toml`` primary keys.

    Args:
        school_config: Loaded :class:`~edvise.configs.genai.SchoolMappingConfig`.
        grain_contracts_by_dataset: Map **dataset name** (same keys as
            ``school_config.datasets``, i.e. inputs.toml table names) to the approved
            :class:`GrainContract` for that table.
        dataset_name_suffix: Same suffix you pass to ``build_schema_contract_from_config``
            (used only to log a warning if ``contract.table`` does not match the logical name).
        canonical_learner_column: Grain primary keys in merged config use this canonical name
            (default ``learner_id`` for GenAI; pass ``student_id`` for legacy audit-style configs).

    Returns:
        New ``SchoolMappingConfig`` with updated ``DatasetConfig.primary_keys`` where provided
        and optional ``cleaning.student_id_alias`` from grain.
    """
    school_config = merge_grain_learner_id_alias_into_school_config(
        school_config, grain_contracts_by_dataset
    )
    unknown = set(grain_contracts_by_dataset) - set(school_config.datasets)
    if unknown:
        raise KeyError(
            "grain_contracts_by_dataset has unknown dataset names (not in school_config): "
            f"{sorted(unknown)}"
        )

    datasets: dict[str, DatasetConfig] = {}
    for name, dc in school_config.datasets.items():
        gc = grain_contracts_by_dataset.get(name)
        if gc is None:
            datasets[name] = dc
            continue

        if gc.institution_id != school_config.institution_id:
            logger.warning(
                "Grain contract institution_id %r != school_config %r for dataset %r",
                gc.institution_id,
                school_config.institution_id,
                name,
            )
        logical = f"{name}{dataset_name_suffix}" if dataset_name_suffix else name
        if gc.table not in (name, logical):
            logger.warning(
                "Grain contract table %r does not match dataset name %r or logical %r",
                gc.table,
                name,
                logical,
            )

        gc_resolved = canonicalize_grain_contract_learner_id_alias(
            gc, canonical_column=canonical_learner_column
        )
        uks = list(gc_resolved.unique_keys)
        if not uks:
            raise ValueError(
                f"Grain contract for dataset {name!r} has empty unique_keys / post_clean_primary_key"
            )
        datasets[name] = dc.model_copy(update={"primary_keys": uks})

    return school_config.model_copy(update={"datasets": datasets})


def dedupe_fn_by_dataset_from_grain_contracts(
    grain_contracts_by_dataset: dict[str, GrainContract],
    *,
    dataset_name_suffix: str = "",
    canonical_learner_column: Literal["student_id", "learner_id"] = "learner_id",
    hook_modules_root: str | Path | None = None,
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """
    Build ``dedupe_fn_by_dataset`` for :func:`~edvise.genai.mapping.schema_contract.build_from_school_config.build_schema_contract_from_config`
    using :func:`~edvise.genai.mapping.identity_agent.execution.contract_utilities.build_dedupe_fn_from_grain_contract`
    per grain contract.

    Keys are **logical** dataset names (same rule as ``build_schema_contract_from_config``):
    ``{dataset_name}{dataset_name_suffix}`` when ``dataset_name_suffix`` is non-empty, else
    ``dataset_name``.

    ``hook_modules_root`` is passed through for contracts with ``dedup_policy.hook_spec`` (materialized
    ``dedup_hooks.py`` under that root).

    :func:`build_schema_contract_from_grain_contracts` applies this automatically and merges
    with any explicit ``dedupe_fn_by_dataset`` (explicit entries override auto-built fns).
    """
    out: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}
    for ds_name, gc in grain_contracts_by_dataset.items():
        logical = f"{ds_name}{dataset_name_suffix}" if dataset_name_suffix else ds_name
        out[logical] = build_dedupe_fn_from_grain_contract(
            gc,
            canonical_learner_column=canonical_learner_column,
            hook_modules_root=hook_modules_root,
        )
    return out


def build_schema_contract_from_grain_contracts(
    school_config: SchoolMappingConfig,
    grain_contracts_by_dataset: dict[str, GrainContract],
    *,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    dataset_name_suffix: str = "",
    sample_size: Optional[int] = None,
    cleaning_cfg: Optional[CleaningConfig] = None,
    term_order_fn: Optional[TermOrderFn] = None,
    term_column_by_dataset: Optional[dict[str, str]] = None,
    term_order_fn_by_dataset: Optional[dict[str, Optional[TermOrderFn]]] = None,
    dedupe_fn_by_dataset: Optional[
        dict[str, Callable[[pd.DataFrame], pd.DataFrame]]
    ] = None,
    canonical_learner_column: Literal["student_id", "learner_id"] = "learner_id",
    hook_modules_root: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build cleaned frames and a frozen schema contract (envelope uses ``student_id_alias`` from
    data audit); primary keys and cleaning alias are taken from grain contracts where provided.

    Applies :func:`merge_grain_contracts_into_school_config` (primary keys + cleaning alias) then
    :func:`edvise.genai.mapping.schema_contract.build_from_school_config.build_schema_contract_from_config`.

    Args:
        school_config: School mapping config (paths, cleaning, baseline primary_keys).
        grain_contracts_by_dataset: Per-dataset grain contracts. Keys are **dataset names**
            matching ``school_config.datasets``.
        term_order_fn: Optional hook passed to
            :func:`~edvise.genai.mapping.schema_mapping_agent.preprocessing.build_schema_contract_from_config`
            (same as ``clean_dataset``). For Identity term-stage output, build from
            :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermOrderConfig` via
            :func:`~edvise.genai.mapping.identity_agent.term_normalization.term_order.term_order_fn_from_term_order_config`.
            When ``term_extraction`` is ``hook_required``, pass ``hook_modules_root=`` (e.g.
            ``school_config.bronze_volumes_path``) so materialized extractors load from ``hook_spec.file``.
        term_column_by_dataset: Logical dataset name → column name for the term-order step.
            When using ``term_order_fn_from_term_order_config``, set each entry to
            ``term_order_column_for_clean_dataset`` for the matching
            :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermOrderConfig`.
        term_order_fn_by_dataset: Optional logical name → term-order hook (per dataset).
            When a key is present, that fn is used; use ``None`` as the value to skip term order
            for that dataset. Datasets not listed fall back to ``term_order_fn``.
        dedupe_fn_by_dataset: Optional per-logical-name overrides for ``CleanSpec.dedupe_fn``.
            If omitted or empty, fns are **auto-built** from each grain contract’s
            ``dedup_policy`` (``sort_by`` / ``keep`` / temporal collapse, custom hook, etc.). Non-empty
            ``dedupe_fn_by_dataset`` is merged on top: **explicit keys replace** the auto fn
            for that dataset (e.g. custom school hooks).
        hook_modules_root: Directory containing ``identity_hooks/`` (e.g. ``school_config.bronze_volumes_path``).
            Used to import ``dedup_policy.hook_spec.file`` when strategy is ``policy_required`` with
            a hook. Defaults to ``school_config.bronze_volumes_path`` when omitted and that path is set.
        dtype_opts, spark_session, sample_size, cleaning_cfg: Forwarded
            to :func:`~edvise.genai.mapping.schema_contract.build_from_school_config.build_schema_contract_from_config`.

    Returns:
        ``(cleaned_dataframes_by_logical_name, schema_contract_dict)`` — same as
        :func:`~edvise.genai.mapping.schema_contract.build_from_school_config.build_schema_contract_from_config`.
    """
    from edvise.genai.mapping.schema_contract.build_from_school_config import (
        build_schema_contract_from_config,
    )

    merged = merge_grain_contracts_into_school_config(
        school_config,
        grain_contracts_by_dataset,
        dataset_name_suffix=dataset_name_suffix,
        canonical_learner_column=canonical_learner_column,
    )
    resolved_hook_root: str | Path | None = hook_modules_root
    if resolved_hook_root is None:
        bv = school_config.bronze_volumes_path
        if bv and str(bv).strip():
            resolved_hook_root = bv
    auto_dedupe = dedupe_fn_by_dataset_from_grain_contracts(
        grain_contracts_by_dataset,
        dataset_name_suffix=dataset_name_suffix,
        canonical_learner_column=canonical_learner_column,
        hook_modules_root=resolved_hook_root,
    )
    merged_dedupe = {**auto_dedupe, **(dedupe_fn_by_dataset or {})}
    return build_schema_contract_from_config(
        merged,
        dtype_opts=dtype_opts,
        spark_session=spark_session,
        dataset_name_suffix=dataset_name_suffix,
        sample_size=sample_size,
        cleaning_cfg=cleaning_cfg,
        term_order_fn=term_order_fn,
        term_column_by_dataset=term_column_by_dataset,
        term_order_fn_by_dataset=term_order_fn_by_dataset,
        dedupe_fn_by_dataset=merged_dedupe if merged_dedupe else None,
        canonical_learner_column=canonical_learner_column,
    )


# --- Training enrichment (historical examples for SMA) ---------------------------------

UNIQUE_VALUES_MAX_CARDINALITY = 50


def _canonical_normalized_column_name(
    norm_col: str,
    learner_id_alias: str | None,
    *,
    canonical_learner_column: str = "learner_id",
) -> str:
    """Match :func:`~edvise.genai.mapping.schema_contract.build_from_school_config._canonical_primary_keys_for_contract` naming."""
    if not learner_id_alias:
        if canonical_learner_column == "learner_id" and norm_col == "student_id":
            return "learner_id"
        return norm_col
    alias_snake = convert_to_snake_case(learner_id_alias)
    if norm_col == alias_snake:
        return canonical_learner_column
    return norm_col


def _df_column_for_column_details(
    norm_col: str,
    df: pd.DataFrame,
    learner_id_alias: str | None,
    *,
    canonical_learner_column: str = "learner_id",
) -> str | None:
    """Resolve header-normalized name to a column present after :func:`~edvise.data_audit.custom_cleaning.clean_dataset`."""
    if norm_col in df.columns:
        return norm_col
    if learner_id_alias:
        alias_snake = convert_to_snake_case(learner_id_alias)
        if norm_col == alias_snake:
            if canonical_learner_column == "learner_id" and "learner_id" in df.columns:
                return "learner_id"
            if "student_id" in df.columns:
                return "student_id"
    return None


def _column_detail_for_df_column(
    df: pd.DataFrame,
    df_col: str,
    *,
    original_name: str,
    normalized_name: str,
    null_counts: pd.Series,
    total_rows: int,
) -> Dict[str, Any]:
    """Build one ``column_details`` row (stats + samples) for a column present on ``df``."""
    series = df[df_col]
    null_count = int(null_counts[df_col])
    null_pct = float(null_count / total_rows * 100) if total_rows > 0 else 0.0
    unique_count = int(series.nunique())

    col_detail: Dict[str, Any] = {
        "original_name": original_name,
        "normalized_name": normalized_name,
        "null_count": null_count,
        "null_percentage": null_pct,
        "unique_count": unique_count,
        "sample_values": [],
    }

    non_null_mask = series.notna()
    if non_null_mask.any():
        col_detail["sample_values"] = [
            str(v) for v in series[non_null_mask].value_counts().head(5).index.tolist()
        ]
        if unique_count <= UNIQUE_VALUES_MAX_CARDINALITY:
            unique_values = sorted(df[df_col].dropna().unique().tolist())
            col_detail["unique_values"] = [str(v) for v in unique_values]

    return col_detail


def _build_column_details(
    df: pd.DataFrame,
    original_columns: list[str],
    column_mapping: dict[str, list[str]],
    *,
    learner_id_alias: str | None = None,
    canonical_learner_column: str = "learner_id",
    frozen_dtypes: dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    orig_to_norm: dict[str, str] = {}
    for norm_col, orig_list in column_mapping.items():
        for orig_col in orig_list:
            orig_to_norm[orig_col] = norm_col

    null_counts = df.isna().sum()
    total_rows = len(df)

    column_details: List[Dict[str, Any]] = []
    profiled_df_cols: set[str] = set()

    for orig_col in original_columns:
        norm_col = orig_to_norm.get(orig_col, orig_col)
        df_col = _df_column_for_column_details(
            norm_col,
            df,
            learner_id_alias,
            canonical_learner_column=canonical_learner_column,
        )
        if df_col is None:
            logger.warning(
                "  Normalized column '%s' not found in DataFrame, skipping", norm_col
            )
            continue

        profiled_df_cols.add(df_col)

        report_norm = _canonical_normalized_column_name(
            norm_col,
            learner_id_alias,
            canonical_learner_column=canonical_learner_column,
        )
        column_details.append(
            _column_detail_for_df_column(
                df,
                df_col,
                original_name=orig_col,
                normalized_name=report_norm,
                null_counts=null_counts,
                total_rows=total_rows,
            )
        )

    # Union: columns on the cleaned frame (and frozen schema dtypes) that are not from the
    # source file — e.g. IdentityAgent term outputs (_edvise_term_season, …).
    dtype_keys = set(frozen_dtypes.keys()) if frozen_dtypes is not None else None
    for df_col in df.columns:
        if df_col in profiled_df_cols:
            continue
        if dtype_keys is not None and df_col not in dtype_keys:
            continue
        column_details.append(
            _column_detail_for_df_column(
                df,
                df_col,
                original_name=df_col,
                normalized_name=_canonical_normalized_column_name(
                    df_col,
                    learner_id_alias,
                    canonical_learner_column=canonical_learner_column,
                ),
                null_counts=null_counts,
                total_rows=total_rows,
            )
        )

    return column_details


def build_training_example_from_schema_contract(
    school_config: SchoolMappingConfig,
    dataset_name: str,
    logical_name: str,
    schema_contract: dict,
    cleaned_dataframes: dict[str, pd.DataFrame],
    original_columns: list[str],
    column_mapping: dict[str, list[str]],
    original_row_count: int,
    file_path: str,
    *,
    canonical_learner_column: str | None = None,
) -> Dict[str, Any]:
    if logical_name not in schema_contract["datasets"]:
        raise KeyError(
            f"Dataset '{logical_name}' not found in schema_contract. "
            f"Available datasets: {list(schema_contract['datasets'].keys())}"
        )

    dataset_schema = schema_contract["datasets"][logical_name]
    df = cleaned_dataframes[logical_name]

    clc = canonical_learner_column or schema_contract.get(
        "canonical_learner_column", "learner_id"
    )
    if clc not in ("student_id", "learner_id"):
        clc = "learner_id"

    sid_alias = (
        school_config.cleaning.student_id_alias if school_config.cleaning else None
    )
    column_details = _build_column_details(
        df=df,
        original_columns=original_columns,
        column_mapping=column_mapping,
        learner_id_alias=sid_alias,
        canonical_learner_column=clc,
        frozen_dtypes=dataset_schema.get("dtypes"),
    )

    orig_to_norm: dict[str, str] = {}
    for norm_col, orig_list in column_mapping.items():
        for orig_col in orig_list:
            orig_to_norm[orig_col] = norm_col

    return {
        "school_id": school_config.institution_id,
        "school_name": school_config.institution_name or school_config.institution_id,
        "dataset_name": dataset_name,
        "file_path": file_path,
        "num_rows": original_row_count,
        "num_columns": len(df.columns),
        "notes": school_config.notes,
        "schema": {
            "normalized_columns": dataset_schema["normalized_columns"],
            "dtypes": dataset_schema["dtypes"],
            "unique_keys": dataset_schema.get("unique_keys", []),
            "non_null_columns": dataset_schema.get("non_null_columns", []),
        },
        "column_normalization": {
            "original_to_normalized": {
                orig: _canonical_normalized_column_name(
                    orig_to_norm.get(orig, orig),
                    sid_alias,
                    canonical_learner_column=clc,
                )
                for orig in original_columns
            },
            "normalized_to_originals": dict(column_mapping),
            "collisions": {
                norm: origs for norm, origs in column_mapping.items() if len(origs) > 1
            },
        },
        "inferred_dtypes": dataset_schema["dtypes"],
        "column_details": column_details,
    }


def process_school_dataset(
    school_config: SchoolMappingConfig,
    dataset_name: str,
    dataset_config: DatasetConfig,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    sample_size: int = 10000,
    dataset_name_suffix: str = "",
    term_order_fn: Optional[TermOrderFn] = None,
    term_column_by_dataset: Optional[dict[str, str]] = None,
    term_order_fn_by_dataset: Optional[dict[str, Optional[TermOrderFn]]] = None,
    grain_contracts_by_dataset: Optional[dict[str, GrainContract]] = None,
    canonical_learner_column: Literal["student_id", "learner_id"] = "learner_id",
) -> tuple[Dict[str, Any], dict]:
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()

    file_path = resolve_genai_data_path(
        school_config.bronze_volumes_path, dataset_config.files[0]
    )

    try:
        from edvise.genai.mapping.schema_contract.build_from_school_config import (
            _load_and_preprocess_dataset,
            build_schema_contract_from_config,
        )

        _, original_columns, column_mapping, original_row_count = (
            _load_and_preprocess_dataset(
                dataset_config=dataset_config,
                spark_session=spark_session,
                sample_size=None,
                bronze_volumes_path=school_config.bronze_volumes_path,
            )
        )

        load_start = time.time()
        partial_school_config = school_config.model_copy(
            update={"datasets": {dataset_name: dataset_config}},
        )
        if grain_contracts_by_dataset:
            cleaned_dataframes, schema_contract = (
                build_schema_contract_from_grain_contracts(
                    school_config=partial_school_config,
                    grain_contracts_by_dataset=grain_contracts_by_dataset,
                    dtype_opts=dtype_opts,
                    spark_session=spark_session,
                    dataset_name_suffix=dataset_name_suffix,
                    sample_size=sample_size,
                    term_order_fn=term_order_fn,
                    term_column_by_dataset=term_column_by_dataset,
                    term_order_fn_by_dataset=term_order_fn_by_dataset,
                    canonical_learner_column=canonical_learner_column,
                )
            )
        else:
            cleaned_dataframes, schema_contract = build_schema_contract_from_config(
                school_config=partial_school_config,
                dtype_opts=dtype_opts,
                spark_session=spark_session,
                dataset_name_suffix=dataset_name_suffix,
                sample_size=sample_size,
                term_order_fn=term_order_fn,
                term_column_by_dataset=term_column_by_dataset,
                term_order_fn_by_dataset=term_order_fn_by_dataset,
                canonical_learner_column=canonical_learner_column,
            )
        logger.debug(
            "  Built schema contract in %.2f seconds", time.time() - load_start
        )

        logical_name = (
            f"{dataset_name}{dataset_name_suffix}"
            if dataset_name_suffix
            else dataset_name
        )

        stats_start = time.time()
        example = build_training_example_from_schema_contract(
            school_config=school_config,
            dataset_name=dataset_name,
            logical_name=logical_name,
            schema_contract=schema_contract,
            cleaned_dataframes=cleaned_dataframes,
            original_columns=original_columns,
            column_mapping=column_mapping,
            original_row_count=original_row_count,
            file_path=file_path,
            canonical_learner_column=canonical_learner_column,
        )

        logger.debug(
            "  Collected column stats in %.2f seconds", time.time() - stats_start
        )

        return example, schema_contract

    except Exception as e:
        logger.error(
            "Error processing %s/%s: %s",
            school_config.institution_id,
            dataset_name,
            e,
            exc_info=True,
        )
        return (
            {
                "school_id": school_config.institution_id,
                "school_name": school_config.institution_name
                or school_config.institution_id,
                "dataset_name": dataset_name,
                "file_path": file_path,
                "error": str(e),
            },
            {},
        )


def build_enriched_schema_contract_for_institution(
    school_config: SchoolMappingConfig,
    *,
    dataset_names: Optional[Sequence[str]] = None,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    sample_size: int = 10_000,
    dataset_name_suffix: str = "",
    term_order_fn: Optional[TermOrderFn] = None,
    term_column_by_dataset: Optional[dict[str, str]] = None,
    term_order_fn_by_dataset: Optional[dict[str, Optional[TermOrderFn]]] = None,
    grain_contracts_by_dataset: Optional[dict[str, GrainContract]] = None,
    canonical_learner_column: Literal["student_id", "learner_id"] = "learner_id",
) -> Dict[str, Any]:
    """
    Build one SMA-style **enriched** institution JSON with **all** requested logical datasets.

    Runs :func:`process_school_dataset` for each name in ``dataset_names`` (or the default set
    described below), then merges schema + per-dataset ``training`` metadata into a single
    institution-style JSON.

    **Default ``dataset_names``:** if ``None``, uses ``sorted(grain_contracts_by_dataset)``
    when ``grain_contracts_by_dataset`` is provided; otherwise ``sorted(school_config.datasets)``.

    For grain-driven primary keys, pass ``grain_contracts_by_dataset``; each dataset processed
    uses ``{dataset_name: grain_contract}`` when that key is present. Pass a ``school_config``
    already updated via :func:`merge_grain_learner_id_alias_into_school_config` with the **full**
    grain map so ``cleaning.student_id_alias`` matches §8 / multi-table workflows.
    """
    if dataset_names is None:
        if grain_contracts_by_dataset:
            names = sorted(grain_contracts_by_dataset.keys())
        else:
            names = sorted(school_config.datasets.keys())
    else:
        names = list(dataset_names)

    if not names:
        raise ValueError("dataset_names is empty (nothing to enrich)")

    school_examples: List[Dict[str, Any]] = []
    schema_contracts: List[tuple[str, dict]] = []
    dataset_to_logical_name: Dict[str, str] = {}

    for dataset_name in names:
        if dataset_name not in school_config.datasets:
            raise KeyError(
                f"dataset_name {dataset_name!r} not in school_config.datasets "
                f"({list(school_config.datasets)})"
            )

        dataset_config = school_config.datasets[dataset_name]
        grain_slice: Optional[dict[str, GrainContract]] = None
        if (
            grain_contracts_by_dataset is not None
            and dataset_name in grain_contracts_by_dataset
        ):
            grain_slice = {dataset_name: grain_contracts_by_dataset[dataset_name]}

        example, schema_contract = process_school_dataset(
            school_config=school_config,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dtype_opts=dtype_opts,
            spark_session=spark_session,
            sample_size=sample_size,
            dataset_name_suffix=dataset_name_suffix,
            term_order_fn=term_order_fn,
            term_column_by_dataset=term_column_by_dataset,
            term_order_fn_by_dataset=term_order_fn_by_dataset,
            grain_contracts_by_dataset=grain_slice,
            canonical_learner_column=canonical_learner_column,
        )

        if "error" in example:
            raise RuntimeError(
                f"Failed to build schema contract for dataset {dataset_name!r}: "
                f"{example['error']}"
            )
        if not schema_contract or "datasets" not in schema_contract:
            raise RuntimeError(
                f"No schema contract produced for dataset {dataset_name!r}"
            )

        for logical_name in schema_contract["datasets"].keys():
            dataset_to_logical_name[dataset_name] = logical_name
            break

        school_examples.append(example)
        schema_contracts.append((dataset_name, schema_contract))

    return _build_enriched_schema_contract(
        school_config=school_config,
        school_examples=school_examples,
        schema_contracts=schema_contracts,
        dataset_to_logical_name=dataset_to_logical_name,
    )


def build_enriched_schema_contract_for_dataset(
    school_config: SchoolMappingConfig,
    dataset_name: str,
    *,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    sample_size: int = 10_000,
    dataset_name_suffix: str = "",
    term_order_fn: Optional[TermOrderFn] = None,
    term_column_by_dataset: Optional[dict[str, str]] = None,
    term_order_fn_by_dataset: Optional[dict[str, Optional[TermOrderFn]]] = None,
    grain_contracts_by_dataset: Optional[dict[str, GrainContract]] = None,
    canonical_learner_column: Literal["student_id", "learner_id"] = "learner_id",
) -> Dict[str, Any]:
    """
    Build one SMA-style **enriched** institution JSON containing a **single** logical dataset.

    Equivalent to :func:`build_enriched_schema_contract_for_institution` with
    ``dataset_names=[dataset_name]``.
    """
    return build_enriched_schema_contract_for_institution(
        school_config,
        dataset_names=[dataset_name],
        dtype_opts=dtype_opts,
        spark_session=spark_session,
        sample_size=sample_size,
        dataset_name_suffix=dataset_name_suffix,
        term_order_fn=term_order_fn,
        term_column_by_dataset=term_column_by_dataset,
        term_order_fn_by_dataset=term_order_fn_by_dataset,
        grain_contracts_by_dataset=grain_contracts_by_dataset,
        canonical_learner_column=canonical_learner_column,
    )


def _build_enriched_schema_contract(
    school_config: SchoolMappingConfig,
    school_examples: List[Dict[str, Any]],
    schema_contracts: List[tuple[str, dict]],
    dataset_to_logical_name: Dict[str, str],
) -> Dict[str, Any]:
    if not schema_contracts:
        raise ValueError("No schema contracts provided")

    merged_datasets: Dict[str, Any] = {}
    for _dataset_name, schema_contract in schema_contracts:
        for logical_name, dataset_schema in schema_contract.get("datasets", {}).items():
            merged_datasets[logical_name] = dataset_schema.copy()

    example_by_dataset = {
        ex["dataset_name"]: ex for ex in school_examples if "error" not in ex
    }

    for logical_name in merged_datasets:
        matching_example = None
        for dataset_name, logical in dataset_to_logical_name.items():
            if logical == logical_name and dataset_name in example_by_dataset:
                matching_example = example_by_dataset[dataset_name]
                break

        if not matching_example:
            for dataset_name, example in example_by_dataset.items():
                if logical_name == dataset_name or logical_name.startswith(
                    dataset_name + "_"
                ):
                    matching_example = example
                    break

        if matching_example:
            merged_datasets[logical_name]["training"] = {
                "file_path": matching_example["file_path"],
                "num_rows": matching_example["num_rows"],
                "num_columns": matching_example["num_columns"],
                "column_normalization": matching_example["column_normalization"],
                "column_details": matching_example["column_details"],
            }

    base_contract = schema_contracts[0][1]
    enriched_contract: Dict[str, Any] = {
        "created_at": base_contract.get("created_at"),
        "null_tokens": base_contract.get("null_tokens", []),
        "school_id": school_config.institution_id,
        "school_name": school_config.institution_name or school_config.institution_id,
        "notes": school_config.notes,
        "datasets": merged_datasets,
    }
    lid = base_contract.get("learner_id_alias") or base_contract.get("student_id_alias")
    if lid:
        enriched_contract["learner_id_alias"] = lid
    clc = base_contract.get("canonical_learner_column")
    if clc:
        enriched_contract["canonical_learner_column"] = clc

    return enriched_contract


def save_enriched_schema_contract(
    enriched_contract: Dict[str, Any],
    output_path: Path,
    *,
    validate_for_sma: bool = False,
) -> None:
    """
    Write a single enriched schema contract JSON (any filename).

    Set ``validate_for_sma=True`` to assert the payload matches
    :class:`~edvise.genai.mapping.schema_contract.EnrichedSchemaContractForSMA`
    before writing.
    """
    if validate_for_sma:
        from edvise.genai.mapping.schema_contract import (
            parse_enriched_schema_contract_for_sma,
        )

        parse_enriched_schema_contract_for_sma(enriched_contract)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(enriched_contract, f, indent=2, default=str)
    logger.info("Saved: %s", output_path.resolve())


def save_enriched_schema_contracts(
    enriched_schema_contracts: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in output_dir.glob("*_example.json"):
        existing_file.unlink()
        logger.info("Removed: %s", existing_file.name)

    for enriched_contract in enriched_schema_contracts:
        school_id = enriched_contract["school_id"]
        filename = f"{school_id}_schema_contract.json"
        filepath = output_dir / filename
        save_enriched_schema_contract(enriched_contract, filepath)

    logger.info("✓ Saved all enriched schema contracts to %s", output_dir)


__all__ = [
    "UNIQUE_VALUES_MAX_CARDINALITY",
    "build_enriched_schema_contract_for_institution",
    "build_enriched_schema_contract_for_dataset",
    "dedupe_fn_by_dataset_from_grain_contracts",
    "build_schema_contract_from_grain_contracts",
    "build_training_example_from_schema_contract",
    "merge_grain_contracts_into_school_config",
    "merge_grain_learner_id_alias_into_school_config",
    "process_school_dataset",
    "save_enriched_schema_contract",
    "save_enriched_schema_contracts",
]
