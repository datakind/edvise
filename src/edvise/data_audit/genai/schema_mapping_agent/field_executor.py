"""
Deterministic join resolver for SchemaMappingAgent pipeline.

Runs after field mapping approval and before transformation map execution.
Infers join graph from:
  1. Approved manifest — which source_tables are referenced per entity type
  2. Schema contract   — unique_keys + column inventory per dataset

Produces a JoinGraph per entity type (cohort / course) which is:
  - Persisted to schema_contract for human review alongside transformation maps
  - Executed deterministically by execute_join_graph() to produce flat input DataFrames

Design:
  - Base table = most granular table (unique keys are superset of other tables' unique keys)
  - Join candidates = all other referenced tables
  - Join keys = intersection of base table canonical columns and candidate canonical unique keys
  - Foreign key fallback = scan all available canonical columns for match to candidate unique keys
  - Fan-out detection = flag when candidate grain is finer than base table grain
  - Join order = topological sort by key dependencies (base table first, then dependents)
  - Alias handling = manifest column_aliases rename source columns to canonical names for
    key matching; original names are preserved for actual DataFrame merge execution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import pandas as pd

from edvise.data_audit.genai.schema_mapping_agent.mapping_schemas import EntityType, FieldMappingManifest

logger = logging.getLogger(__name__)

# Threshold for using Spark instead of pandas (rows)
SPARK_THRESHOLD = 500000


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class JoinStep:
    """A single join operation in the join graph."""
    left: str                           # Left table name (running DataFrame after prior steps)
    right: str                          # Right table name
    left_on: list[str]                  # Join keys on left (original column names)
    right_on: list[str]                 # Join keys on right (original column names, may differ)
    how: str = "left"                   # Join type — always left for safety
    fan_out_risk: bool = False          # True if right table grain is finer than left
    fan_out_note: Optional[str] = None  # Human-readable explanation of fan-out risk


@dataclass
class JoinGraph:
    """
    Complete join graph for one entity type.
    Produced by JoinResolver.resolve(), reviewed by human, executed by execute_join_graph().
    """
    entity_type: EntityType
    base_table: str
    steps: list[JoinStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # Ambiguities needing human review

    @property
    def referenced_tables(self) -> list[str]:
        return [self.base_table] + [s.right for s in self.steps]

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "base_table": self.base_table,
            "steps": [
                {
                    "left": s.left,
                    "right": s.right,
                    "left_on": s.left_on,
                    "right_on": s.right_on,
                    "how": s.how,
                    "fan_out_risk": s.fan_out_risk,
                    "fan_out_note": s.fan_out_note,
                }
                for s in self.steps
            ],
            "warnings": self.warnings,
        }


# -----------------------------------------------------------------------------
# Join graph executor
# -----------------------------------------------------------------------------

def execute_join_graph(
    graph: JoinGraph,
    dataframes: dict[str, pd.DataFrame],
    spark_session: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Execute an approved JoinGraph against loaded DataFrames.

    For large datasets (>500k rows), automatically uses Spark if available to avoid
    memory issues. Falls back to optimized pandas for smaller datasets or when
    Spark is not available.

    Args:
        graph: Approved JoinGraph from JoinResolver.resolve()
        dataframes: Dict of dataset_name -> DataFrame.
                    Keys must match graph.base_table and step.right values.
        spark_session: Optional Spark session. If provided and base table is large,
                      will use Spark for joins to avoid memory issues.

    Returns:
        Flat input DataFrame ready for TransformationMap executor.
    """
    if graph.base_table not in dataframes:
        raise ValueError(
            f"Base table '{graph.base_table}' not found in dataframes. "
            f"Available: {list(dataframes.keys())}"
        )

    base_df = dataframes[graph.base_table]
    base_rows = len(base_df)
    
    # Determine if we should use Spark
    use_spark = (
        spark_session is not None
        and base_rows > SPARK_THRESHOLD
    )
    
    if use_spark:
        logger.info(
            f"[{graph.entity_type}] Using Spark for large dataset "
            f"({base_rows:,} rows) from base '{graph.base_table}'"
        )
        return _execute_join_graph_spark(graph, dataframes, spark_session)
    else:
        logger.info(
            f"[{graph.entity_type}] Using pandas for join from base '{graph.base_table}': "
            f"{base_df.shape}"
        )
        return _execute_join_graph_pandas(graph, dataframes)


def _execute_join_graph_pandas(
    graph: JoinGraph,
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Execute join graph using pandas (optimized for memory).
    
    Uses in-place operations where possible to reduce memory footprint.
    """
    if graph.base_table not in dataframes:
        raise ValueError(
            f"Base table '{graph.base_table}' not found in dataframes. "
            f"Available: {list(dataframes.keys())}"
        )

    # Use copy only if we need to preserve original (which we do)
    # But we'll be more careful about memory in the merge loop
    result = dataframes[graph.base_table].copy()
    logger.info(
        f"[{graph.entity_type}] Starting join from base '{graph.base_table}': "
        f"{result.shape}"
    )

    for step in graph.steps:
        if step.right not in dataframes:
            raise ValueError(
                f"Table '{step.right}' not found in dataframes. "
                f"Available: {list(dataframes.keys())}"
            )

        right_df = dataframes[step.right]

        if step.fan_out_risk:
            logger.warning(
                f"Executing join with fan-out risk: {step.left} ← {step.right} "
                f"on {step.left_on} = {step.right_on}. {step.fan_out_note}"
            )

        pre_shape = result.shape
        # Use merge with explicit memory-efficient options
        result = result.merge(
            right_df,
            left_on=step.left_on,
            right_on=step.right_on,
            how=step.how,
            suffixes=("", f"_{step.right}"),
            copy=False,  # Try to avoid copying when possible
        )

        # Drop duplicate columns introduced by the join.
        # Keep left (base) side columns over right side suffixed duplicates.
        # This handles cases where both tables share column names like student_id.
        dupes = result.columns[result.columns.duplicated(keep="first")].tolist()
        if dupes:
            logger.debug(
                f"Dropping duplicate columns after joining '{step.right}': {dupes}"
            )
            result = result.loc[:, ~result.columns.duplicated(keep="first")]

        logger.info(
            f"Joined '{step.right}' on {step.left_on} = {step.right_on}: "
            f"{pre_shape} → {result.shape}"
        )

    return result


def _execute_join_graph_spark(
    graph: JoinGraph,
    dataframes: dict[str, pd.DataFrame],
    spark_session: Any,
) -> pd.DataFrame:
    """
    Execute join graph using Spark DataFrames for large datasets.
    
    Converts pandas DataFrames to Spark, performs joins, then converts back.
    This avoids memory issues with large datasets in pandas.
    """
    try:
        # Convert base table to Spark DataFrame
        base_df = dataframes[graph.base_table]
        result_spark = spark_session.createDataFrame(base_df)
        
        logger.info(
            f"[{graph.entity_type}] Starting Spark join from base '{graph.base_table}': "
            f"{base_df.shape}"
        )
        
        for step in graph.steps:
            if step.right not in dataframes:
                raise ValueError(
                    f"Table '{step.right}' not found in dataframes. "
                    f"Available: {list(dataframes.keys())}"
                )
            
            right_df = dataframes[step.right]
            right_spark = spark_session.createDataFrame(right_df)
            
            if step.fan_out_risk:
                logger.warning(
                    f"Executing Spark join with fan-out risk: {step.left} ← {step.right} "
                    f"on {step.left_on} = {step.right_on}. {step.fan_out_note}"
                )
            
            pre_count = result_spark.count()
            pre_cols = len(result_spark.columns)
            
            # Rename right table columns to avoid conflicts (matching pandas behavior)
            # Add suffix to all right table columns except join keys
            result_cols = set(result_spark.columns)
            right_rename = {}
            
            for col in right_spark.columns:
                if col not in step.right_on:
                    # Check if this column name exists in result (potential conflict)
                    if col in result_cols:
                        # Conflict - add suffix like pandas does
                        right_rename[col] = f"{col}_{step.right}"
            
            # Apply renames
            if right_rename:
                for old_name, new_name in right_rename.items():
                    right_spark = right_spark.withColumnRenamed(old_name, new_name)
            
            # Perform join using Spark DataFrame API
            # Build join condition using column expressions
            from pyspark.sql.functions import col
            
            join_expr = None
            for left_col, right_col in zip(step.left_on, step.right_on):
                # Use result_spark and right_spark column references
                condition = result_spark[left_col] == right_spark[right_col]
                if join_expr is None:
                    join_expr = condition
                else:
                    join_expr = join_expr & condition
            
            # Perform the join
            result_spark = result_spark.join(
                right_spark,
                join_expr,
                how=step.how,
            )
            
            # Note: Spark may keep duplicate join key columns. When we convert to pandas,
            # pandas will automatically handle duplicate column names by adding suffixes.
            # This matches the pandas merge behavior with suffixes.
            
            post_count = result_spark.count()
            post_cols = len(result_spark.columns)
            logger.info(
                f"Joined '{step.right}' on {step.left_on} = {step.right_on}: "
                f"({pre_count}, {pre_cols}) → ({post_count}, {post_cols})"
            )
        
        # Convert back to pandas
        logger.info(f"[{graph.entity_type}] Converting Spark result back to pandas...")
        result = result_spark.toPandas()
        
        # Drop duplicate columns introduced by Spark joins
        # Spark keeps both left and right join key columns when they have the same name
        # This matches the pandas merge behavior where we keep left side columns
        dupes = result.columns[result.columns.duplicated(keep="first")].tolist()
        if dupes:
            logger.debug(
                f"Dropping duplicate columns after Spark join: {dupes}"
            )
            result = result.loc[:, ~result.columns.duplicated(keep="first")]
        
        logger.info(f"[{graph.entity_type}] Final result shape: {result.shape}")
        
        return result
        
    except Exception as e:
        logger.error(
            f"Error in Spark join execution: {e}. "
            f"Falling back to pandas (this may cause memory issues with large datasets)."
        )
        import traceback
        logger.debug(traceback.format_exc())
        # Fallback to pandas if Spark fails
        return _execute_join_graph_pandas(graph, dataframes)