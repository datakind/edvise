"""
Preprocessing step for Agent 2 pipeline (Milestone 1).

Builds schema contract from inputs.toml config by:
1. Loading raw files from config
2. Normalizing column names
3. Generating training dtypes
4. Building schema contract with unique keys from config

This produces the normalized DataFrame + schema_contract.json needed as input to Agent 2.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from edvise.data_audit.custom_cleaning import (
    DtypeGenerationOptions,
    SchemaContractMeta,
    SchemaFreezeOptions,
    build_schema_contract,
    freeze_schema,
    generate_training_dtypes,
    normalize_columns,
)
from edvise.dataio.read import from_csv_file
from edvise.configs.genai import DatasetConfig, SchoolMappingConfig

logger = logging.getLogger(__name__)


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
) -> tuple[dict[str, pd.DataFrame], dict]:
    """
    Build schema contract from inputs.toml SchoolMappingConfig.
    
    This is the Milestone 1 preprocessing step that:
    1. Loads raw files from config
    2. Normalizes column names using normalize_columns()
    3. Generates training dtypes using generate_training_dtypes()
    4. Builds schema contract with unique keys from config
    
    Args:
        school_config: SchoolMappingConfig from inputs.toml
        dtype_opts: Optional DtypeGenerationOptions for dtype inference
        spark_session: Optional Spark session for reading files
        dataset_name_suffix: Suffix to add to dataset names (default: "" uses config names as-is).
                            Pass "_df" to add suffix (e.g., "student" -> "student_df").
        sample_size: Optional max rows to sample per dataset (None = use all data).
                    Useful for PoC/testing to speed up processing.
    
    Returns:
        Tuple of (cleaned_dataframes, schema_contract):
        - cleaned_dataframes: Dict mapping dataset_name -> cleaned DataFrame
        - schema_contract: Schema contract dict ready for JoinResolver
    """
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()
    
    cleaned_map: dict[str, pd.DataFrame] = {}
    specs: dict[str, dict[str, Any]] = {}
    
    logger.info(
        "Building schema contract for %s (%s)",
        school_config.institution_name or school_config.institution_id,
        school_config.institution_id,
    )
    
    for dataset_name, dataset_config in school_config.datasets.items():
        # Use dataset name from config, optionally with suffix
        logical_name = f"{dataset_name}{dataset_name_suffix}" if dataset_name_suffix else dataset_name
        
        logger.info("Processing dataset: %s (logical name: %s)", dataset_name, logical_name)
        
        # Use shared preprocessing function
        df_with_dtypes, original_columns, column_mapping, original_row_count = _load_and_preprocess_dataset(
            dataset_config=dataset_config,
            dtype_opts=dtype_opts,
            spark_session=spark_session,
            sample_size=sample_size,  # Allow sampling for PoC/testing
        )
        
        logger.info("  Raw shape: %s", df_with_dtypes.shape)
        
        # Check for collisions
        collisions = {norm: origs for norm, origs in column_mapping.items() if len(origs) > 1}
        if collisions:
            logger.warning(
                "  Column name collisions after normalization: %s",
                collisions,
            )
        
        # Store cleaned DataFrame
        cleaned_map[logical_name] = df_with_dtypes
        
        # Build spec for schema contract
        # Unique keys come from config, but need to be normalized
        unique_keys = dataset_config.primary_keys or []  # Config still uses primary_keys field name
        # Normalize unique key names to match normalized columns
        # column_mapping is {normalized_name: [original_names]}
        # Create a reverse mapping: original -> normalized
        orig_to_norm = {}
        for norm_col, orig_list in column_mapping.items():
            for orig_col in orig_list:
                orig_to_norm[orig_col] = norm_col
        
        # Get normalized column names from DataFrame (already normalized)
        normalized_columns = set(df_with_dtypes.columns)
        
        normalized_uks = []
        for uk in unique_keys:
            # Check if unique key is already normalized (exact match in normalized columns)
            if uk in normalized_columns:
                normalized_uks.append(uk)
            elif uk in orig_to_norm:
                # Unique key is an original column name, get its normalized version
                normalized_uks.append(orig_to_norm[uk])
            else:
                # Unique key not found - try normalizing it directly
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
            "non-null columns": [],  # Can be populated if needed
            "_orig_cols_": original_columns,
        }
        
        logger.info(
            "  ✓ Processed: %d rows, %d columns, unique keys = %s",
            len(df_with_dtypes),
            len(df_with_dtypes.columns),
            normalized_uks,
        )
    
    # Build schema contract
    meta = SchemaContractMeta(
        created_at=datetime.now(timezone.utc).isoformat(),
        null_tokens=["(Blank)"],  # Default null tokens
    )
    freeze_opts = SchemaFreezeOptions(include_column_order_hash=True)
    
    schema_contract = build_schema_contract(
        cleaned_map=cleaned_map,
        specs=specs,
        meta=meta,
        freeze_opts=freeze_opts,
    )
    
    # Schema contract uses "unique_keys" consistently
    
    logger.info(
        "Schema contract built with %d datasets: %s",
        len(schema_contract["datasets"]),
        list(schema_contract["datasets"].keys()),
    )
    
    return cleaned_map, schema_contract
