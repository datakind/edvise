"""
Helper functions for processing school datasets and building historical examples.

This module provides utilities for:
- Processing raw school data files
- Normalizing column names
- Inferring data types
- Building structured examples for GenAI training

This module now uses schema_contract as the base and enriches it with training-specific
metadata (sample values, null stats, unique values for low cardinality columns). Column
dtypes for prompts and enforcement live only in each dataset's frozen ``dtypes`` map.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

import pandas as pd

from edvise.data_audit.custom_cleaning import DtypeGenerationOptions
from edvise.configs.genai import SchoolMappingConfig, DatasetConfig

LOGGER = logging.getLogger(__name__)

# Maximum cardinality for capturing all unique values
# Columns with <= this many unique values will have their complete unique value list captured
UNIQUE_VALUES_MAX_CARDINALITY = 50


def _build_column_details(
    df: pd.DataFrame,
    original_columns: list[str],
    column_mapping: dict[str, list[str]],
) -> List[Dict[str, Any]]:
    """
    Build detailed column information enriched with training metadata.
    
    Args:
        df: DataFrame with normalized columns (used for null stats and samples)
        original_columns: List of original column names (before normalization)
        column_mapping: Dict {normalized_name: [original_names]}
        
    Returns:
        List of column detail dictionaries with training metadata
    """
    # Build reverse mapping for original_to_normalized
    orig_to_norm = {}
    for norm_col, orig_list in column_mapping.items():
        for orig_col in orig_list:
            orig_to_norm[orig_col] = norm_col
    
    # Pre-calculate null counts once for all columns (optimization)
    null_counts = df.isna().sum()
    total_rows = len(df)

    column_details = []
    for orig_col in original_columns:
        norm_col = orig_to_norm.get(orig_col, orig_col)
        if norm_col not in df.columns:
            LOGGER.warning("  Normalized column '%s' not found in DataFrame, skipping", norm_col)
            continue
            
        series = df[norm_col]
        
        # Use pre-calculated null count (avoid redundant isna().sum() calls)
        null_count = int(null_counts[norm_col])
        null_pct = (
            float(null_count / total_rows * 100) if total_rows > 0 else 0.0
        )
        
        unique_count = int(series.nunique())
        
        col_detail = {
            "original_name": orig_col,
            "normalized_name": norm_col,
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
            "sample_values": [],
        }
        
        # More efficient sample collection - get most frequent sample values
        non_null_mask = series.notna()
        if non_null_mask.any():
            # Get up to 5 most frequent values (more representative than first unique)
            col_detail["sample_values"] = [
                str(v) for v in series[non_null_mask].value_counts().head(5).index.tolist()
            ]
            
            # Capture all unique values for low cardinality columns
            # This is critical for SchemaMappingAgent to see all possible values (e.g., all grades at UCF)
            if unique_count <= UNIQUE_VALUES_MAX_CARDINALITY:
                unique_values = sorted(
                    df[norm_col].dropna().unique().tolist()
                )
                # Convert to strings for JSON serialization
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
    """
    Build training example by enriching schema_contract with training metadata.
    
    Uses schema_contract as the base and adds:
    - Sample values per column
    - Null statistics
    - Unique counts
    - Complete unique values for low cardinality columns (<= 50 unique values)
    - Column details
    
    Args:
        school_config: SchoolMappingConfig with school metadata
        dataset_name: Name of the dataset (e.g., "student", "course")
        logical_name: Logical dataset name (may have suffix like "_df")
        schema_contract: Schema contract dict from build_schema_contract_from_config
        cleaned_dataframes: Dict of cleaned DataFrames from build_schema_contract_from_config
        original_columns: List of original column names (before normalization)
        column_mapping: Dict {normalized_name: [original_names]}
        original_row_count: Original row count before sampling
        file_path: Path to source file
        
    Returns:
        Dictionary with training example structure
    """
    # Extract schema for this dataset from contract
    if logical_name not in schema_contract["datasets"]:
        raise KeyError(
            f"Dataset '{logical_name}' not found in schema_contract. "
            f"Available datasets: {list(schema_contract['datasets'].keys())}"
        )
    
    dataset_schema = schema_contract["datasets"][logical_name]
    df = cleaned_dataframes[logical_name]
    
    # Build column details with training metadata
    column_details = _build_column_details(
        df=df,
        original_columns=original_columns,
        column_mapping=column_mapping,
    )
    
    # Build reverse mapping for original_to_normalized
    orig_to_norm = {}
    for norm_col, orig_list in column_mapping.items():
        for orig_col in orig_list:
            orig_to_norm[orig_col] = norm_col
    
    # Build the example structure
    example = {
        "school_id": school_config.institution_id,
        "school_name": school_config.institution_name or school_config.institution_id,
        "dataset_name": dataset_name,
        "file_path": file_path,
        "num_rows": original_row_count,  # Store original count, not sampled
        "num_columns": len(original_columns),
        "notes": school_config.notes,
        # Schema info from contract
        "schema": {
            "normalized_columns": dataset_schema["normalized_columns"],
            "dtypes": dataset_schema["dtypes"],
            "unique_keys": dataset_schema.get("unique_keys", []),
            "non_null_columns": dataset_schema.get("non_null_columns", []),
        },
        # Column normalization info (for backward compatibility)
        "column_normalization": {
            "original_to_normalized": {
                orig: orig_to_norm.get(orig, orig) for orig in original_columns
            },
            "normalized_to_originals": dict(column_mapping),
            "collisions": {
                norm: origs
                for norm, origs in column_mapping.items()
                if len(origs) > 1
            },
        },
        # Legacy field for backward compatibility
        "inferred_dtypes": dataset_schema["dtypes"],
        # Training-specific enrichment
        "column_details": column_details,
    }
    
    return example


def process_school_dataset(
    school_config: SchoolMappingConfig,
    dataset_name: str,
    dataset_config: DatasetConfig,
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    sample_size: int = 10000,
    dataset_name_suffix: str = "",
    term_order_fn: Optional[Any] = None,
    term_col_by_dataset: Optional[dict[str, str]] = None,
) -> tuple[Dict[str, Any], dict]:
    """
    Process a single dataset for a school using schema_contract as the base.
    
    This function:
    1. Builds schema_contract from config (using build_schema_contract_from_config)
    2. Enriches it with training-specific metadata (sample values, null stats, unique values)
    3. Returns structured example for GenAI training and schema_contract
    
    Args:
        school_config: SchoolMappingConfig with school metadata
        dataset_name: Name of the dataset (e.g., "student", "course")
        dataset_config: DatasetConfig with file paths
        dtype_opts: Optional DtypeGenerationOptions for dtype inference
        spark_session: Optional Spark session for reading files
        sample_size: Maximum number of rows to process (for performance)
        dataset_name_suffix: Suffix to add to dataset names (default: "").
                            Pass "_df" to add suffix (e.g., "student" -> "student_df").
        term_order_fn: Optional callable (df, term_col) -> df that adds a term_order column.
        term_col_by_dataset: Optional dict mapping logical_name -> term column name.
        
    Returns:
        Tuple of (example_dict, schema_contract):
        - example_dict: Dictionary with normalization mapping, inferred dtypes, and training metadata
        - schema_contract: Schema contract dictionary for this dataset
    """
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()
    
    # Use the first file for file_path in error reporting
    file_path = dataset_config.files[0]
    
    try:
        # Get original row count before sampling by doing a quick preprocessing pass
        # This is needed to report accurate row counts in training examples
        from edvise.genai.schema_mapping_agent.preprocessing import (
            _load_and_preprocess_dataset,
            build_schema_contract_from_config,
        )
        
        # Get original row count and column info (without sampling)
        _, original_columns, column_mapping, original_row_count = _load_and_preprocess_dataset(
            dataset_config=dataset_config,
            dtype_opts=dtype_opts,
            spark_session=spark_session,
            sample_size=None,  # Get full count
        )
        
        # Now build schema contract with sampling for performance
        load_start = time.time()
        partial_school_config = school_config.model_copy(
            update={"datasets": {dataset_name: dataset_config}},
        )
        cleaned_dataframes, schema_contract = build_schema_contract_from_config(
            school_config=partial_school_config,
            dtype_opts=dtype_opts,
            spark_session=spark_session,
            dataset_name_suffix=dataset_name_suffix,
            sample_size=sample_size,
            term_order_fn=term_order_fn,
            term_col_by_dataset=term_col_by_dataset,
        )
        LOGGER.debug("  Built schema contract in %.2f seconds", time.time() - load_start)
        
        # Get logical name (may have suffix)
        logical_name = f"{dataset_name}{dataset_name_suffix}" if dataset_name_suffix else dataset_name
        
        # Build training example from schema contract
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
        
        LOGGER.debug("  Collected column stats in %.2f seconds", time.time() - stats_start)
        
        return example, schema_contract
        
    except Exception as e:
        LOGGER.error(
            "Error processing %s/%s: %s",
            school_config.institution_id,
            dataset_name,
            e,
            exc_info=True,
        )
        error_example = {
            "school_id": school_config.institution_id,
            "school_name": school_config.institution_name or school_config.institution_id,
            "dataset_name": dataset_name,
            "file_path": file_path,
            "error": str(e),
        }
        return error_example, {}


def process_all_schools(
    project_config: Any,  # MappingProjectConfig
    dtype_opts: Optional[DtypeGenerationOptions] = None,
    spark_session: Optional[Any] = None,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Process all schools and datasets from a MappingProjectConfig.
    
    Args:
        project_config: MappingProjectConfig with all schools
        dtype_opts: Optional DtypeGenerationOptions for dtype inference
        spark_session: Optional Spark session for reading files
        output_dir: Optional directory to save schema_contract.json files (if None, doesn't save)
        
    Returns:
        List of enriched schema_contract dictionaries, one per institution
    """
    if dtype_opts is None:
        dtype_opts = DtypeGenerationOptions()
    
    all_examples = []
    enriched_schema_contracts = []
    total_start = time.time()
    
    for school_key, school_config in project_config.schools.items():
        school_start = time.time()
        LOGGER.info("Processing: %s (%s)", school_config.institution_name or school_key, school_key)
        
        # Collect all examples and schema contracts for this school
        school_examples = []
        school_schema_contracts = []
        dataset_to_logical_name = {}  # Track mapping from dataset_name to logical_name
        
        for dataset_name, dataset_config in school_config.datasets.items():
            dataset_start = time.time()
            LOGGER.info("  Dataset: %s (%d files)", dataset_name, len(dataset_config.files))
            
            example, schema_contract = process_school_dataset(
                school_config=school_config,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                dtype_opts=dtype_opts,
                spark_session=spark_session,
            )
            
            # Track the logical_name for this dataset
            if schema_contract and "datasets" in schema_contract:
                for logical_name in schema_contract["datasets"].keys():
                    dataset_to_logical_name[dataset_name] = logical_name
                    break  # Should only be one dataset per contract
            
            school_examples.append(example)
            if schema_contract:
                school_schema_contracts.append((dataset_name, schema_contract))
            
            dataset_elapsed = time.time() - dataset_start
            if "error" not in example:
                LOGGER.info(
                    "    ✓ Processed: %d rows, %d columns (%.2f seconds)",
                    example["num_rows"],
                    example["num_columns"],
                    dataset_elapsed,
                )
                if example.get("column_normalization", {}).get("collisions"):
                    LOGGER.warning(
                        "    ⚠ Column name collisions: %d",
                        len(example["column_normalization"]["collisions"]),
                    )
            else:
                LOGGER.error("    ✗ Error: %s (%.2f seconds)", example["error"], dataset_elapsed)
        
        # Build enriched schema_contract for this school
        if school_schema_contracts:
            enriched_contract = _build_enriched_schema_contract(
                school_config=school_config,
                school_examples=school_examples,
                schema_contracts=school_schema_contracts,
                dataset_to_logical_name=dataset_to_logical_name,
            )
            enriched_schema_contracts.append(enriched_contract)
        
        all_examples.extend(school_examples)
        
        school_elapsed = time.time() - school_start
        LOGGER.info("  School %s completed in %.2f seconds", school_key, school_elapsed)
    
    total_elapsed = time.time() - total_start
    LOGGER.info("Total processing time: %.2f seconds", total_elapsed)
    
    # Save enriched schema_contracts if output_dir is provided
    if output_dir is not None:
        save_enriched_schema_contracts(enriched_schema_contracts, output_dir)
    
    return enriched_schema_contracts


def _build_enriched_schema_contract(
    school_config: SchoolMappingConfig,
    school_examples: List[Dict[str, Any]],
    schema_contracts: List[tuple[str, dict]],
    dataset_to_logical_name: Dict[str, str],
) -> Dict[str, Any]:
    """
    Build enriched schema_contract by merging training examples into schema contract.
    
    Args:
        school_config: SchoolMappingConfig with school metadata
        school_examples: List of training example dictionaries for this school
        schema_contracts: List of (dataset_name, schema_contract) tuples for this school
        dataset_to_logical_name: Dict mapping dataset_name to logical_name
        
    Returns:
        Enriched schema_contract dictionary with training metadata merged in
    """
    # Start with the first schema_contract as base (they should all have same top-level structure)
    if not schema_contracts:
        raise ValueError("No schema contracts provided")
    
    # Merge all datasets from all schema contracts
    merged_datasets = {}
    for dataset_name, schema_contract in schema_contracts:
        # Get logical name (may have suffix)
        for logical_name, dataset_schema in schema_contract.get("datasets", {}).items():
            merged_datasets[logical_name] = dataset_schema.copy()
    
    # Find matching examples and merge training metadata
    example_by_dataset = {ex["dataset_name"]: ex for ex in school_examples if "error" not in ex}
    
    for logical_name in merged_datasets:
        # Find the example that matches this dataset using the mapping
        matching_example = None
        for dataset_name, logical in dataset_to_logical_name.items():
            if logical == logical_name and dataset_name in example_by_dataset:
                matching_example = example_by_dataset[dataset_name]
                break
        
        # Fallback: try direct matching if mapping didn't work
        if not matching_example:
            for dataset_name, example in example_by_dataset.items():
                # Check if this example's logical name matches
                # The logical name might be dataset_name or dataset_name + suffix
                if logical_name == dataset_name or logical_name.startswith(dataset_name + "_"):
                    matching_example = example
                    break
        
        if matching_example:
            # Add training metadata to this dataset
            merged_datasets[logical_name]["training"] = {
                "file_path": matching_example["file_path"],
                "num_rows": matching_example["num_rows"],
                "num_columns": matching_example["num_columns"],
                "column_normalization": matching_example["column_normalization"],
                "column_details": matching_example["column_details"],
            }
    
    # Build enriched schema contract
    base_contract = schema_contracts[0][1]  # Use first contract as base
    enriched_contract = {
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


def save_enriched_schema_contracts(
    enriched_schema_contracts: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Save enriched schema_contract.json files, one per institution.
    
    Args:
        enriched_schema_contracts: List of enriched schema contract dictionaries
        output_dir: Directory to save files
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any existing example files
    for existing_file in output_dir.glob("*_example.json"):
        existing_file.unlink()
        LOGGER.info("Removed: %s", existing_file.name)
    
    # Save one schema_contract.json per institution
    for enriched_contract in enriched_schema_contracts:
        school_id = enriched_contract["school_id"]
        filename = f"{school_id}_schema_contract.json"
        filepath = output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(enriched_contract, f, indent=2, default=str)
        
        LOGGER.info("Saved: %s", filename)
    
    LOGGER.info("✓ Saved all enriched schema contracts to %s", output_dir)
