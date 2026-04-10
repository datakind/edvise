"""
Grain merge, frozen schema-contract build, and enriched training payloads for GenAI.

Combines :func:`merge_grain_contracts_into_school_config` / :func:`build_schema_contract_from_grain_contracts`
with helpers that attach historical-example metadata (samples, null stats, low-cardinality uniques)
for Schema Mapping Agent prompts.

The **canonical JSON shape** consumed by Schema Mapping Agent is
:class:`~edvise.genai.mapping.schema_contract.EnrichedSchemaContractForSMA`
(enriched institution file with ``school_id`` and per-dataset ``training``).

Use :func:`build_enriched_schema_contract_for_dataset` once per logical dataset, then
:func:`save_enriched_schema_contract` (or loop and save with your naming convention).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from edvise.configs.custom import CleaningConfig
from edvise.configs.genai import DatasetConfig, SchoolMappingConfig
from edvise.data_audit.custom_cleaning import DtypeGenerationOptions, TermOrderFn
from edvise.genai.mapping.identity_agent.execution.contract_utilities import (
    canonicalize_grain_contract_student_id_alias,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.utils.data_cleaning import convert_to_snake_case

logger = logging.getLogger(__name__)


def merge_grain_student_id_alias_into_school_config(
    school_config: SchoolMappingConfig,
    grain_contracts_by_dataset: dict[str, GrainContract],
) -> SchoolMappingConfig:
    """
    Set ``school_config.cleaning.student_id_alias`` from grain when contracts specify it.

    All non-null ``student_id_alias`` values across the given contracts must agree; otherwise
    raises. When ``inputs.toml`` already sets a different alias, grain wins and a warning is logged.

    Callers that build schema contracts **one dataset at a time** (e.g. manifest helpers) should
    run this once per institution with the **full** grain map so ``cleaning`` matches every table,
    then pass a single-dataset slice into :func:`merge_grain_contracts_into_school_config` /
    :func:`build_schema_contract_from_grain_contracts` for primary-key overrides.
    """
    aliases: set[str] = set()
    for gc in grain_contracts_by_dataset.values():
        if gc.student_id_alias:
            aliases.add(gc.student_id_alias.strip())
    if not aliases:
        return school_config
    if len(aliases) > 1:
        raise ValueError(
            "Grain contracts disagree on student_id_alias: "
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

    Returns:
        New ``SchoolMappingConfig`` with updated ``DatasetConfig.primary_keys`` where provided
        and optional ``cleaning.student_id_alias`` from grain.
    """
    school_config = merge_grain_student_id_alias_into_school_config(
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

        gc_resolved = canonicalize_grain_contract_student_id_alias(gc)
        uks = list(gc_resolved.unique_keys)
        if not uks:
            raise ValueError(
                f"Grain contract for dataset {name!r} has empty unique_keys / post_clean_primary_key"
            )
        datasets[name] = dc.model_copy(update={"primary_keys": uks})

    return school_config.model_copy(update={"datasets": datasets})


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
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build cleaned frames and a frozen schema contract, with primary keys and ``student_id_alias``
    taken from grain contracts where provided.

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
        term_column_by_dataset: Logical dataset name → column name for the term-order step.
            When using ``term_order_fn_from_term_order_config``, set each entry to
            ``term_order_column_for_clean_dataset`` for the matching
            :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermOrderConfig`.
        term_order_fn_by_dataset: Optional logical name → term-order hook (per dataset).
            When a key is present, that fn is used; use ``None`` as the value to skip term order
            for that dataset. Datasets not listed fall back to ``term_order_fn``.
        dedupe_fn_by_dataset, dtype_opts, spark_session, sample_size, cleaning_cfg: Forwarded
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
    )
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
        dedupe_fn_by_dataset=dedupe_fn_by_dataset,
    )


# --- Training enrichment (historical examples for SMA) ---------------------------------

UNIQUE_VALUES_MAX_CARDINALITY = 50


def _canonical_normalized_column_name(
    norm_col: str,
    student_id_alias: str | None,
) -> str:
    """Match :func:`~edvise.genai.mapping.schema_contract.build_from_school_config._canonical_primary_keys_for_contract` naming."""
    if not student_id_alias:
        return norm_col
    alias_snake = convert_to_snake_case(student_id_alias)
    if norm_col == alias_snake:
        return "student_id"
    return norm_col


def _df_column_for_column_details(
    norm_col: str,
    df: pd.DataFrame,
    student_id_alias: str | None,
) -> str | None:
    """Resolve header-normalized name to a column present after :func:`~edvise.data_audit.custom_cleaning.clean_dataset`."""
    if norm_col in df.columns:
        return norm_col
    if student_id_alias:
        alias_snake = convert_to_snake_case(student_id_alias)
        if norm_col == alias_snake and "student_id" in df.columns:
            return "student_id"
    return None


def _build_column_details(
    df: pd.DataFrame,
    original_columns: list[str],
    column_mapping: dict[str, list[str]],
    *,
    student_id_alias: str | None = None,
) -> List[Dict[str, Any]]:
    orig_to_norm: dict[str, str] = {}
    for norm_col, orig_list in column_mapping.items():
        for orig_col in orig_list:
            orig_to_norm[orig_col] = norm_col

    null_counts = df.isna().sum()
    total_rows = len(df)

    column_details: List[Dict[str, Any]] = []
    for orig_col in original_columns:
        norm_col = orig_to_norm.get(orig_col, orig_col)
        df_col = _df_column_for_column_details(norm_col, df, student_id_alias)
        if df_col is None:
            logger.warning(
                "  Normalized column '%s' not found in DataFrame, skipping", norm_col
            )
            continue

        series = df[df_col]
        null_count = int(null_counts[df_col])
        null_pct = float(null_count / total_rows * 100) if total_rows > 0 else 0.0
        unique_count = int(series.nunique())

        report_norm = _canonical_normalized_column_name(norm_col, student_id_alias)
        col_detail: Dict[str, Any] = {
            "original_name": orig_col,
            "normalized_name": report_norm,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "sample_values": [],
        }

        non_null_mask = series.notna()
        if non_null_mask.any():
            col_detail["sample_values"] = [
                str(v)
                for v in series[non_null_mask].value_counts().head(5).index.tolist()
            ]
            if unique_count <= UNIQUE_VALUES_MAX_CARDINALITY:
                unique_values = sorted(df[df_col].dropna().unique().tolist())
                col_detail["unique_values"] = [str(v) for v in unique_values]

        column_details.append(col_detail)

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
) -> Dict[str, Any]:
    if logical_name not in schema_contract["datasets"]:
        raise KeyError(
            f"Dataset '{logical_name}' not found in schema_contract. "
            f"Available datasets: {list(schema_contract['datasets'].keys())}"
        )

    dataset_schema = schema_contract["datasets"][logical_name]
    df = cleaned_dataframes[logical_name]

    sid_alias = (
        school_config.cleaning.student_id_alias if school_config.cleaning else None
    )
    column_details = _build_column_details(
        df=df,
        original_columns=original_columns,
        column_mapping=column_mapping,
        student_id_alias=sid_alias,
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
        "num_columns": len(original_columns),
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
                    orig_to_norm.get(orig, orig), sid_alias
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
) -> tuple[Dict[str, Any], dict]:
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()

    file_path = dataset_config.files[0]

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
) -> Dict[str, Any]:
    """
    Build one SMA-style **enriched** institution JSON containing a **single** logical dataset.

    Runs :func:`process_school_dataset` for ``dataset_name``, then merges schema + ``training``
    metadata into one institution-style JSON (single dataset under ``datasets``).

    For grain-driven primary keys, pass ``grain_contracts_by_dataset`` with at least
    ``{dataset_name: grain_contract}``. To align ``student_id_alias`` with other tables,
    pass a ``school_config`` already updated via
    :func:`merge_grain_student_id_alias_into_school_config` using the **full** grain map.
    """
    if dataset_name not in school_config.datasets:
        raise KeyError(
            f"dataset_name {dataset_name!r} not in school_config.datasets "
            f"({list(school_config.datasets)})"
        )

    dataset_config = school_config.datasets[dataset_name]
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
        grain_contracts_by_dataset=grain_contracts_by_dataset,
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

    dataset_to_logical_name: Dict[str, str] = {}
    for logical_name in schema_contract["datasets"].keys():
        dataset_to_logical_name[dataset_name] = logical_name
        break

    return _build_enriched_schema_contract(
        school_config=school_config,
        school_examples=[example],
        schema_contracts=[(dataset_name, schema_contract)],
        dataset_to_logical_name=dataset_to_logical_name,
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
    if base_contract.get("student_id_alias"):
        enriched_contract["student_id_alias"] = base_contract["student_id_alias"]

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
        from edvise.genai.mapping.schema_contract import parse_enriched_schema_contract_for_sma

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
    "build_enriched_schema_contract_for_dataset",
    "build_schema_contract_from_grain_contracts",
    "build_training_example_from_schema_contract",
    "merge_grain_contracts_into_school_config",
    "merge_grain_student_id_alias_into_school_config",
    "process_school_dataset",
    "save_enriched_schema_contract",
    "save_enriched_schema_contracts",
]
