"""
NSC SFTP ingestion helpers.

NSC-specific utilities for processing SFTP files, extracting institution IDs,
managing ingestion manifests, and working with Databricks schemas/volumes.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import pyspark.sql

from edvise.utils.api_requests import databricksify_inst_name
from edvise.utils.data_cleaning import convert_to_snake_case

LOGGER = logging.getLogger(__name__)

# Schema and volume caches
_schema_cache: dict[str, set[str]] = {}
_bronze_volume_cache: dict[str, str] = {}  # key: f"{catalog}.{schema}" -> volume_name


def ensure_plan_table(spark: pyspark.sql.SparkSession, plan_table: str) -> None:
    """
    Create institution_ingest_plan table if it doesn't exist.

    Args:
        spark: Spark session
        plan_table: Full table path (e.g., "catalog.schema.table")
    """
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {plan_table} (
          file_fingerprint STRING,
          file_name STRING,
          local_path STRING,
          institution_id STRING,
          inst_col STRING,
          file_size BIGINT,
          file_modified_time TIMESTAMP,
          planned_at TIMESTAMP
        )
        USING DELTA
        """
    )


def detect_institution_column(cols: list[str], inst_col_pattern: re.Pattern) -> Optional[str]:
    """
    Detect institution ID column using regex pattern.

    Args:
        cols: List of column names
        inst_col_pattern: Compiled regex pattern to match institution column

    Returns:
        Matched column name or None if not found

    Example:
        >>> pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
        >>> detect_institution_column(["student_id", "institution_id"], pattern)
        'institution_id'
    """
    return next((c for c in cols if inst_col_pattern.search(c)), None)


def extract_institution_ids(
    local_path: str,
    *,
    renames: dict[str, str],
    inst_col_pattern: re.Pattern,
) -> tuple[Optional[str], list[str]]:
    """
    Extract unique institution IDs from a staged CSV file.

    Reads file, normalizes/renames columns, detects institution column,
    and returns unique institution IDs.

    Args:
        local_path: Path to local CSV file
        renames: Dictionary mapping old column names to new names
        inst_col_pattern: Compiled regex pattern to match institution column

    Returns:
        Tuple of (institution_column_name, sorted_list_of_unique_ids).
        Returns (None, []) if no institution column found.

    Example:
        >>> pattern = re.compile(r"(?=.*institution)(?=.*id)", re.IGNORECASE)
        >>> renames = {"inst_id": "institution_id"}
        >>> col, ids = extract_institution_ids(
        ...     "/tmp/file.csv", renames=renames, inst_col_pattern=pattern
        ... )
        >>> print(col, ids)
        'institution_id' ['12345', '67890']
    """
    df = pd.read_csv(local_path, on_bad_lines="warn")
    # Use convert_to_snake_case from utils instead of normalize_col
    df = df.rename(columns={c: convert_to_snake_case(c) for c in df.columns})
    df = df.rename(columns=renames)

    inst_col = detect_institution_column(df.columns.tolist(), inst_col_pattern)
    if inst_col is None:
        return None, []

    # Make IDs robust: drop nulls, strip whitespace, keep as string
    series = df[inst_col].dropna()

    # Some files store as numeric; normalize to integer-like strings when possible
    ids = set()
    for v in series.tolist():
        # Handle pandas/numpy numeric types
        try:
            if isinstance(v, int):
                ids.add(str(v))
                continue
            if isinstance(v, float):
                # If 323100.0 -> "323100"
                if v.is_integer():
                    ids.add(str(int(v)))
                else:
                    ids.add(str(v).strip())
                continue
        except Exception:
            pass

        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            continue
        # If it's "323100.0" as string, coerce safely
        if re.fullmatch(r"\d+\.0+", s):
            s = s.split(".")[0]
        ids.add(s)

    return inst_col, sorted(ids)


def output_file_name_from_sftp(file_name: str) -> str:
    """
    Generate output filename from SFTP filename.

    Removes extension and adds .csv extension.

    Args:
        file_name: Original SFTP filename

    Returns:
        Output filename with .csv extension

    Example:
        >>> output_file_name_from_sftp("data_2024.xlsx")
        'data_2024.csv'
    """
    return f"{os.path.basename(file_name).split('.')[0]}.csv"


def list_schemas_in_catalog(spark: pyspark.sql.SparkSession, catalog: str) -> set[str]:
    """
    List all schemas in a catalog (with caching).

    Args:
        spark: Spark session
        catalog: Catalog name

    Returns:
        Set of schema names
    """
    if catalog in _schema_cache:
        return _schema_cache[catalog]

    rows = spark.sql(f"SHOW SCHEMAS IN {catalog}").collect()

    schema_names: set[str] = set()
    for row in rows:
        d = row.asDict()
        for k in ["databaseName", "database_name", "schemaName", "schema_name", "name"]:
            v = d.get(k)
            if v:
                schema_names.add(v)
                break
        else:
            schema_names.add(list(d.values())[0])

    _schema_cache[catalog] = schema_names
    return schema_names


def find_bronze_schema(
    spark: pyspark.sql.SparkSession, catalog: str, inst_prefix: str
) -> str:
    """
    Find bronze schema for institution prefix.

    Args:
        spark: Spark session
        catalog: Catalog name
        inst_prefix: Institution prefix (e.g., "motlow_state_cc")

    Returns:
        Bronze schema name (e.g., "motlow_state_cc_bronze")

    Raises:
        ValueError: If bronze schema not found
    """
    target = f"{inst_prefix}_bronze"
    schemas = list_schemas_in_catalog(spark, catalog)
    if target not in schemas:
        raise ValueError(f"Bronze schema not found: {catalog}.{target}")
    return target


def find_bronze_volume_name(
    spark: pyspark.sql.SparkSession, catalog: str, schema: str
) -> str:
    """
    Find bronze volume name in schema (with caching).

    Args:
        spark: Spark session
        catalog: Catalog name
        schema: Schema name

    Returns:
        Volume name containing "bronze"

    Raises:
        ValueError: If no bronze volume found
    """
    key = f"{catalog}.{schema}"
    if key in _bronze_volume_cache:
        return _bronze_volume_cache[key]

    vols = spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}").collect()
    if not vols:
        raise ValueError(f"No volumes found in {catalog}.{schema}")

    # Usually "volume_name", but be defensive
    def _get_vol_name(row):
        d = row.asDict()
        for k in ["volume_name", "volumeName", "name"]:
            if k in d:
                return d[k]
        return list(d.values())[0]

    vol_names = [_get_vol_name(v) for v in vols]
    bronze_like = [v for v in vol_names if "bronze" in str(v).lower()]
    if bronze_like:
        _bronze_volume_cache[key] = bronze_like[0]
        return bronze_like[0]

    raise ValueError(
        f"No volume containing 'bronze' found in {catalog}.{schema}. Volumes={vol_names}"
    )


def update_manifest(
    spark: pyspark.sql.SparkSession,
    manifest_table: str,
    file_fingerprint: str,
    *,
    status: str,
    error_message: Optional[str],
) -> None:
    """
    Update ingestion_manifest for a file_fingerprint.

    Assumes upstream inserted status=NEW already. Updates status, error_message,
    and timestamps.

    Args:
        spark: Spark session
        manifest_table: Full table path (e.g., "catalog.schema.table")
        file_fingerprint: File fingerprint identifier
        status: New status (e.g., "BRONZE_WRITTEN", "FAILED")
        error_message: Error message if status is FAILED, None otherwise
    """
    from pyspark.sql import types as T

    now_ts = datetime.now(timezone.utc)

    # ingested_at only set when we finish BRONZE_WRITTEN
    row = {
        "file_fingerprint": file_fingerprint,
        "status": status,
        "error_message": error_message,
        "ingested_at": now_ts if status == "BRONZE_WRITTEN" else None,
        "processed_at": now_ts,
    }

    schema = T.StructType(
        [
            T.StructField("file_fingerprint", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("error_message", T.StringType(), True),
            T.StructField("ingested_at", T.TimestampType(), True),
            T.StructField("processed_at", T.TimestampType(), False),
        ]
    )
    df = spark.createDataFrame([row], schema=schema)
    df.createOrReplaceTempView("manifest_updates")

    spark.sql(
        f"""
        MERGE INTO {manifest_table} AS t
        USING manifest_updates AS s
        ON t.file_fingerprint = s.file_fingerprint
        WHEN MATCHED THEN UPDATE SET
          t.status = s.status,
          t.error_message = s.error_message,
          t.ingested_at = COALESCE(s.ingested_at, t.ingested_at),
          t.processed_at = s.processed_at
        """
    )


def process_and_save_file(
    volume_dir: str, file_name: str, df: pd.DataFrame
) -> str:
    """
    Process DataFrame and save to Databricks volume.

    Normalizes column names and saves as CSV.

    Args:
        volume_dir: Volume directory path
        file_name: Output filename
        df: DataFrame to save

    Returns:
        Full path to saved file
    """
    local_file_path = os.path.join(volume_dir, file_name)

    LOGGER.info(f"Saving to Volumes {local_file_path}")
    # Normalize column names for Databricks compatibility
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", col) for col in df.columns]
    df.to_csv(local_file_path, index=False)
    LOGGER.info(f"Saved {file_name} to {local_file_path}")

    return local_file_path
