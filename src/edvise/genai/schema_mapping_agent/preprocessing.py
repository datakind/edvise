"""
Preprocessing step for SchemaMappingAgent pipeline (Milestone 1).

After loading (and optional row sampling) from config paths, each dataset is cleaned with
:class:`edvise.data_audit.custom_cleaning.clean_dataset` — same steps as the custom audit
pipeline: normalize headers, student-id alias rename, null tokens, optional column drops /
non-null row drops, training dtypes, dedupe hook, full-row dedupe, primary-key dedupe and
uniqueness check, then term order. This keeps GenAI schema contracts and training examples
aligned with ``clean_dataset``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, Optional

import pandas as pd

from edvise.data_audit.custom_cleaning import (
    DtypeGenerationOptions,
    SchemaContractMeta,
    SchemaFreezeOptions,
    TermOrderFn,
    build_schema_contract,
    clean_dataset,
    dtype_opts_from_cleaning_config,
    normalize_columns,
)
from edvise.dataio.read import from_csv_file
from edvise.utils.data_cleaning import convert_to_snake_case
from edvise.configs.custom import CleaningConfig
from edvise.configs.genai import DatasetConfig, SchoolMappingConfig

logger = logging.getLogger(__name__)


def _merged_cleaning_cfg(
    school_config: SchoolMappingConfig,
    cleaning_cfg: Optional[CleaningConfig],
) -> Optional[CleaningConfig]:
    """Explicit ``cleaning_cfg`` wins; else ``school_config.cleaning``."""
    if cleaning_cfg is not None:
        return cleaning_cfg
    return school_config.cleaning


def _load_and_preprocess_dataset(
    dataset_config: DatasetConfig,
    spark_session: Optional[Any] = None,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], int]:
    """
    Load CSV(s), optionally sample rows, return raw frame plus column metadata.

    Cleaning is performed separately via :func:`clean_dataset` so it matches the
    custom-school pipeline.

    Returns:
        Tuple of (df_raw, original_columns, column_mapping, original_row_count):
        - df_raw: Loaded data (original column names, before ``clean_dataset``)
        - original_columns: Names as in the file(s)
        - column_mapping: ``normalize_columns`` mapping for metadata / examples
        - original_row_count: Rows before sampling (for reporting)
    """
    dfs = []
    for file_path in dataset_config.files:
        df = from_csv_file(file_path, spark_session=spark_session)
        dfs.append(df)

    if len(dfs) > 1:
        df_raw = pd.concat(dfs, ignore_index=True)
        logger.debug("  Combined %d files into %d rows", len(dfs), len(df_raw))
    else:
        df_raw = dfs[0]

    original_row_count = len(df_raw)

    if sample_size is not None and len(df_raw) > sample_size:
        df_raw = df_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(
            "  Sampled %d rows from %d total rows for faster processing",
            sample_size,
            original_row_count,
        )

    original_columns = list(df_raw.columns)
    _, column_mapping = normalize_columns(df_raw.columns)

    return df_raw, original_columns, column_mapping, original_row_count


def _resolve_primary_keys_to_normalized(
    column_mapping: dict[str, list[str]],
    unique_keys: list[str],
    logical_name: str,
) -> list[str]:
    """Map configured primary key names to normalized column names (pre-``clean_dataset``)."""
    normalized_columns = set(column_mapping.keys())
    orig_to_norm: dict[str, str] = {}
    for norm_col, orig_list in column_mapping.items():
        for orig_col in orig_list:
            orig_to_norm[orig_col] = norm_col

    normalized_uks: list[str] = []
    for uk in unique_keys:
        if uk in normalized_columns:
            normalized_uks.append(uk)
        elif uk in orig_to_norm:
            normalized_uks.append(orig_to_norm[uk])
        else:
            normalized_uk = convert_to_snake_case(uk)
            if normalized_uk in normalized_columns:
                normalized_uks.append(normalized_uk)
            else:
                logger.warning(
                    "  Unique key '%s' (normalized: '%s') not found in normalized "
                    "columns for %s, skipping",
                    uk,
                    normalized_uk,
                    logical_name,
                )
    return normalized_uks


def _primary_keys_for_column_resolution(
    primary_keys: list[str],
    student_id_alias: str | None,
) -> list[str]:
    """
    Map canonical ``student_id`` to the normalized on-disk column name when
    ``CleaningConfig.student_id_alias`` is set.

    ``primary_keys`` in inputs.toml use logical ``student_id``; the CSV column is
    named ``student_id_alias`` until :func:`clean_dataset` renames it. Resolution
    must match :func:`normalize_columns` keys (e.g. ``student_id_randomized_datakind``).
    """
    if not student_id_alias:
        return list(primary_keys)
    alias_snake = convert_to_snake_case(student_id_alias)
    return [alias_snake if k == "student_id" else k for k in primary_keys]


def _canonical_primary_keys_for_contract(
    primary_keys: list[str],
    student_id_alias: str | None,
) -> list[str]:
    """
    Primary key column names as they appear after ``clean_dataset`` (canonical ``student_id``).

    Used for ``freeze_schema`` / schema contract JSON so contracts match inputs.toml and
    cleaned dataframe columns, not the pre-rename alias name.
    """
    if not student_id_alias:
        return list(primary_keys)
    alias_snake = convert_to_snake_case(student_id_alias)
    out: list[str] = []
    for k in primary_keys:
        if k in ("student_id", student_id_alias, alias_snake):
            out.append("student_id")
        else:
            out.append(k)
    return out


def build_schema_contract_from_config(
    school_config: SchoolMappingConfig,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    dataset_name_suffix: str = "",
    sample_size: Optional[int] = None,
    cleaning_cfg: Optional[CleaningConfig] = None,
    term_order_fn: Optional[TermOrderFn] = None,
    term_col_by_dataset: Optional[dict[str, str]] = None,
    dedupe_fn_by_dataset: Optional[
        dict[str, Callable[[pd.DataFrame], pd.DataFrame]]
    ] = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build schema contract from inputs.toml SchoolMappingConfig.

    Each dataset is cleaned with :func:`clean_dataset` (same order as custom audit).

    Args:
        school_config: SchoolMappingConfig from inputs.toml
        dtype_opts: Optional DtypeGenerationOptions; merged with cleaning-based dtype options
        spark_session: Optional Spark session for reading files
        dataset_name_suffix: Suffix for logical dataset names (e.g. ``"_df"``).
        sample_size: Optional max rows per dataset after load (``None`` = all rows).
        term_order_fn: Passed through ``CleanSpec.term_order_fn`` (runs last in ``clean_dataset``).
        term_col_by_dataset: Logical name -> term column name (default ``"term"``).
        dedupe_fn_by_dataset: Logical name -> dedupe hook (``CleanSpec.dedupe_fn``, after dtypes).
        cleaning_cfg: Overrides / supplements ``school_config.cleaning`` for ``clean_dataset``.

    Returns:
        Tuple of (cleaned_dataframes, schema_contract)
    """
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()

    term_col_by_dataset = term_col_by_dataset or {}
    dedupe_fn_by_dataset = dedupe_fn_by_dataset or {}
    merged_cleaning = _merged_cleaning_cfg(school_config, cleaning_cfg)
    cfg_dtype_opts = dtype_opts_from_cleaning_config(merged_cleaning)
    dtype_opts = replace(
        dtype_opts,
        forced_dtypes={**cfg_dtype_opts.forced_dtypes, **dtype_opts.forced_dtypes},
    )

    cleaned_map: dict[str, pd.DataFrame] = {}
    specs: dict[str, dict[str, Any]] = {}

    logger.info(
        "Building schema contract for %s (%s)",
        school_config.institution_name or school_config.institution_id,
        school_config.institution_id,
    )

    for dataset_name, dataset_config in school_config.datasets.items():
        logical_name = (
            f"{dataset_name}{dataset_name_suffix}"
            if dataset_name_suffix
            else dataset_name
        )

        logger.info(
            "Processing dataset: %s (logical name: %s)", dataset_name, logical_name
        )

        df_raw, original_columns, column_mapping, original_row_count = (
            _load_and_preprocess_dataset(
                dataset_config=dataset_config,
                spark_session=spark_session,
                sample_size=sample_size,
            )
        )

        term_col = term_col_by_dataset.get(logical_name, "term")

        pk_config = list(dataset_config.primary_keys or [])
        pk_for_resolution = _primary_keys_for_column_resolution(
            pk_config,
            merged_cleaning.student_id_alias if merged_cleaning else None,
        )
        normalized_uks = _resolve_primary_keys_to_normalized(
            column_mapping, pk_for_resolution, logical_name
        )
        unique_keys_for_contract = _canonical_primary_keys_for_contract(
            pk_config,
            merged_cleaning.student_id_alias if merged_cleaning else None,
        )

        clean_spec: dict[str, Any] = {
            "unique keys": normalized_uks,
            "non-null columns": [],
            "drop columns": None,
            "_orig_cols_": original_columns,
            "term_column": term_col,
            "dedupe_fn": dedupe_fn_by_dataset.get(logical_name),
            "term_order_fn": term_order_fn,
        }

        df_clean = clean_dataset(
            df_raw,
            clean_spec,
            dataset_name=logical_name,
            inference_opts=dtype_opts,
            enforce_uniqueness=True,
            generate_dtypes=True,
            cleaning_cfg=merged_cleaning,
        )

        logger.info(
            "  ✓ After clean_dataset: %d rows, %d columns, unique keys = %s "
            "(raw rows before sample=%d)",
            len(df_clean),
            len(df_clean.columns),
            unique_keys_for_contract,
            original_row_count,
        )

        cleaned_map[logical_name] = df_clean
        # Contract stores canonical names (student_id), not pre-alias column names;
        # clean_spec above uses normalized_uks for on-disk resolution only.
        specs[logical_name] = {
            "unique keys": unique_keys_for_contract,
            "non-null columns": [],
            "_orig_cols_": original_columns,
            "term_column": term_col,
        }

    contract_null_tokens = (
        list(merged_cleaning.null_tokens) if merged_cleaning else ["(Blank)"]
    )
    meta = SchemaContractMeta(
        created_at=datetime.now(timezone.utc).isoformat(),
        null_tokens=contract_null_tokens,
        student_id_alias=(
            merged_cleaning.student_id_alias if merged_cleaning else None
        ),
    )
    freeze_opts = SchemaFreezeOptions(include_column_order_hash=True)

    schema_contract = build_schema_contract(
        cleaned_map=cleaned_map,
        specs=specs,
        meta=meta,
        freeze_opts=freeze_opts,
    )

    logger.info(
        "Schema contract built with %d datasets: %s",
        len(schema_contract["datasets"]),
        list(schema_contract["datasets"].keys()),
    )

    return cleaned_map, schema_contract
