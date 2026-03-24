"""
Preprocessing step for SchemaMappingAgent pipeline (Milestone 1).

Builds schema contract from inputs.toml config by:
1. Loading raw files from config
2. Normalizing column names
3. Generating training dtypes
4. Optionally renaming via ``CleaningConfig.student_id_alias`` using
   ``rename_student_id_alias_column`` (same helper as ``clean_dataset``)
5. Asserting configured primary keys are unique on each dataset (when all keys resolve)
6. Building schema contract with unique keys from config

This produces the normalized DataFrame + schema_contract.json needed as input to SchemaMappingAgent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any, Optional

import pandas as pd

from edvise.data_audit.custom_cleaning import (
    DtypeGenerationOptions,
    SchemaContractMeta,
    SchemaFreezeOptions,
    TermOrderFn,
    assert_dataframe_unique_keys,
    build_schema_contract,
    generate_training_dtypes,
    normalize_columns,
    rename_student_id_alias_column,
)
from edvise.dataio.read import from_csv_file
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
    dtype_opts: DtypeGenerationOptions,
    spark_session: Optional[Any] = None,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], int]:
    """
    Shared preprocessing logic: load files, normalize columns, generate dtypes.
    
    This is the single source of truth for dataset preprocessing used by both:
    - build_schema_contract_from_config (for schema contract building)
    - process_school_dataset (for historical example generation)
    
    Args:
        dataset_config: DatasetConfig with file paths
        dtype_opts: DtypeGenerationOptions for dtype inference
        spark_session: Optional Spark session for reading files
        sample_size: Optional max rows to sample (None = use all data)
                    Used for historical examples to speed up processing
    
    Returns:
        Tuple of (cleaned_df, original_columns, column_mapping, original_row_count):
        - cleaned_df: DataFrame with normalized columns and inferred dtypes
        - original_columns: List of original column names (before normalization)
        - column_mapping: Dict {normalized_name: [original_names]} for collision detection
        - original_row_count: Row count before sampling (for reporting)
    """
    # Load all files for this dataset
    dfs = []
    for file_path in dataset_config.files:
        df = from_csv_file(file_path, spark_session=spark_session)
        dfs.append(df)
    
    # Combine multiple files if needed
    if len(dfs) > 1:
        df_raw = pd.concat(dfs, ignore_index=True)
        logger.debug("  Combined %d files into %d rows", len(dfs), len(df_raw))
    else:
        df_raw = dfs[0]
    
    # Store original row count before any sampling
    original_row_count = len(df_raw)
    
    # Sample if requested (for historical examples generation)
    if sample_size is not None and len(df_raw) > sample_size:
        df_raw = df_raw.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(
            "  Sampled %d rows from %d total rows for faster processing",
            sample_size,
            original_row_count,
        )
    
    # Store original columns
    original_columns = list(df_raw.columns)
    
    # Normalize column names
    normalized_index, column_mapping = normalize_columns(df_raw.columns)
    df_normalized = df_raw.copy()
    df_normalized.columns = normalized_index
    
    # Generate training dtypes
    df_with_dtypes = generate_training_dtypes(df_normalized, opts=dtype_opts)
    
    return df_with_dtypes, original_columns, column_mapping, original_row_count


def build_schema_contract_from_config(
    school_config: SchoolMappingConfig,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    dataset_name_suffix: str = "",
    sample_size: Optional[int] = None,
    cleaning_cfg: Optional[CleaningConfig] = None,
    # --- term order ---
    term_order_fn: Optional[TermOrderFn] = None,
    term_col_by_dataset: Optional[dict[str, str]] = None,
    dedupe_fn_by_dataset: Optional[dict[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build schema contract from inputs.toml SchoolMappingConfig.

    Args:
        school_config: SchoolMappingConfig from inputs.toml
        dtype_opts: Optional DtypeGenerationOptions for dtype inference
        spark_session: Optional Spark session for reading files
        dataset_name_suffix: Suffix to add to dataset names (default: "" uses config names as-is).
                            Pass "_df" to add suffix (e.g., "student" -> "student_df").
        sample_size: Optional max rows to sample per dataset (None = use all data).
        term_order_fn: Optional callable (df, term_col) -> df that adds a term_order column.
                       Typically edvise.feature_generation.term.add_term_order.
                       Applied after dtype generation for datasets that have a term column.
        term_col_by_dataset: Optional dict mapping logical_name -> term column name.
                             e.g. {"student_df": "term_desc", "course_df": "term_descr"}
                             If term_order_fn is provided but a dataset is not in this dict,
                             the default term column "term" is tried.
        dedupe_fn_by_dataset: Optional mapping logical dataset name (e.g. ``course_df``) to
            ``(DataFrame) -> DataFrame``. Applied after term order and student-id alias rename,
            before unique-key checks — same stage as ``CleanSpec.dedupe_fn`` in custom cleaning.
        cleaning_cfg: When set, ``student_id_alias`` is taken from this object only.
                      When omitted, uses ``school_config.cleaning.student_id_alias`` if present.

    Returns:
        Tuple of (cleaned_dataframes, schema_contract)
    """
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()

    term_col_by_dataset = term_col_by_dataset or {}
    dedupe_fn_by_dataset = dedupe_fn_by_dataset or {}
    merged_cleaning = _merged_cleaning_cfg(school_config, cleaning_cfg)

    cleaned_map: dict[str, pd.DataFrame] = {}
    specs: dict[str, dict[str, Any]] = {}

    logger.info(
        "Building schema contract for %s (%s)",
        school_config.institution_name or school_config.institution_id,
        school_config.institution_id,
    )

    for dataset_name, dataset_config in school_config.datasets.items():
        logical_name = f"{dataset_name}{dataset_name_suffix}" if dataset_name_suffix else dataset_name

        logger.info("Processing dataset: %s (logical name: %s)", dataset_name, logical_name)

        df_with_dtypes, original_columns, column_mapping, original_row_count = _load_and_preprocess_dataset(
            dataset_config=dataset_config,
            dtype_opts=dtype_opts,
            spark_session=spark_session,
            sample_size=sample_size,
        )

        logger.info("  Raw shape: %s", df_with_dtypes.shape)

        # Check for collisions
        collisions = {norm: origs for norm, origs in column_mapping.items() if len(origs) > 1}
        if collisions:
            logger.warning("  Column name collisions after normalization: %s", collisions)

        # --- Determine term column (for spec and optional term_order_fn) ---
        term_col = term_col_by_dataset.get(logical_name, "term")

        # --- Apply term order if provided ---
        if term_order_fn is not None:
            if term_col in df_with_dtypes.columns:
                try:
                    df_with_dtypes = term_order_fn(df_with_dtypes, term_col)
                    logger.info(
                        "  Applied term_order_fn on column '%s' → added term_order",
                        term_col,
                    )
                except Exception as e:
                    logger.warning(
                        "  term_order_fn failed for dataset '%s' on column '%s': %s",
                        logical_name,
                        term_col,
                        e,
                    )
            else:
                logger.debug(
                    "  term_order_fn skipped for '%s' — term column '%s' not found",
                    logical_name,
                    term_col,
                )

        if merged_cleaning and merged_cleaning.student_id_alias:
            df_with_dtypes, _ = rename_student_id_alias_column(
                df_with_dtypes,
                merged_cleaning.student_id_alias,
                dataset_label=logical_name,
            )

        if logical_name in dedupe_fn_by_dataset:
            fn = dedupe_fn_by_dataset[logical_name]
            df_with_dtypes = fn(df_with_dtypes)
            logger.info(
                "  Applied dedupe_fn_by_dataset['%s'] → shape=%s",
                logical_name,
                df_with_dtypes.shape,
            )

        cleaned_map[logical_name] = df_with_dtypes

        # Build spec — unique keys normalized (sync with student_id rename when alias used in PK list)
        pk_list = list(dataset_config.primary_keys or [])
        if merged_cleaning and merged_cleaning.student_id_alias:
            pk_list = [
                "student_id" if k == merged_cleaning.student_id_alias else k
                for k in pk_list
            ]
        unique_keys = pk_list
        orig_to_norm = {}
        for norm_col, orig_list in column_mapping.items():
            for orig_col in orig_list:
                orig_to_norm[orig_col] = norm_col

        normalized_columns = set(df_with_dtypes.columns)
        normalized_uks = []
        for uk in unique_keys:
            if uk in normalized_columns:
                normalized_uks.append(uk)
            elif uk in orig_to_norm:
                normalized_uks.append(orig_to_norm[uk])
            else:
                from edvise.utils.data_cleaning import convert_to_snake_case
                normalized_uk = convert_to_snake_case(uk)
                if normalized_uk in normalized_columns:
                    normalized_uks.append(normalized_uk)
                else:
                    logger.warning(
                        "  Unique key '%s' (normalized: '%s') not found in normalized columns, skipping",
                        uk,
                        normalized_uk,
                    )

        specs[logical_name] = {
            "unique keys": normalized_uks,
            "non-null columns": [],
            "_orig_cols_": original_columns,
            "term_column": term_col,
        }

        logger.info(
            "  ✓ Processed: %d rows, %d columns, unique keys = %s",
            len(df_with_dtypes),
            len(df_with_dtypes.columns),
            normalized_uks,
        )

        if len(normalized_uks) != len(unique_keys):
            logger.warning(
                "  Skipping unique-key check for %s: resolved %s from configured keys %s",
                logical_name,
                normalized_uks,
                unique_keys,
            )
        else:
            assert_dataframe_unique_keys(
                df_with_dtypes,
                normalized_uks,
                dataset_name=logical_name,
            )

    # Build schema contract
    meta = SchemaContractMeta(
        created_at=datetime.now(timezone.utc).isoformat(),
        null_tokens=["(Blank)"],
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