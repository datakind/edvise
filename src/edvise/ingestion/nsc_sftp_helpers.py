"""
NSC SFTP ingestion helpers.

NSC-specific utilities for processing SFTP files, extracting institution IDs,
managing ingestion manifests, and working with Databricks schemas/volumes.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import pandas as pd
import pyspark.sql

if TYPE_CHECKING:
    import paramiko
from pyspark.sql import functions as F
from pyspark.sql import types as T

from edvise.ingestion.constants import (
    MANIFEST_TABLE_PATH,
    QUEUE_TABLE_PATH,
    SFTP_DOWNLOAD_CHUNK_MB,
    SFTP_TMP_DIR,
)
from edvise.utils.data_cleaning import convert_to_snake_case, detect_institution_column
from edvise.utils.sftp import download_sftp_atomic

LOGGER = logging.getLogger(__name__)


def ensure_manifest_and_queue_tables(spark: pyspark.sql.SparkSession) -> None:
    """
    Create required delta tables if missing.
    - ingestion_manifest: includes file_fingerprint for idempotency
    - pending_ingest_queue: holds local tmp path so downstream doesn't connect to SFTP again

    Args:
        spark: Spark session
    """
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {MANIFEST_TABLE_PATH} (
          file_fingerprint STRING,
          source_system STRING,
          sftp_path STRING,
          file_name STRING,
          file_size BIGINT,
          file_modified_time TIMESTAMP,
          ingested_at TIMESTAMP,
          processed_at TIMESTAMP,
          status STRING,
          error_message STRING
        )
        USING DELTA
        """
    )

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {QUEUE_TABLE_PATH} (
          file_fingerprint STRING,
          source_system STRING,
          sftp_path STRING,
          file_name STRING,
          file_size BIGINT,
          file_modified_time TIMESTAMP,
          local_tmp_path STRING,
          queued_at TIMESTAMP
        )
        USING DELTA
        """
    )


def build_listing_df(
    spark: pyspark.sql.SparkSession, file_rows: list[dict]
) -> pyspark.sql.DataFrame:
    """
    Build DataFrame from file listing rows with file fingerprints.

    Creates a DataFrame with file metadata and computes a stable fingerprint
    from metadata (file version identity).

    Args:
        spark: Spark session
        file_rows: List of dicts with keys: source_system, sftp_path, file_name,
                   file_size, file_modified_time

    Returns:
        DataFrame with file_fingerprint column added
    """
    schema = T.StructType(
        [
            T.StructField("source_system", T.StringType(), False),
            T.StructField("sftp_path", T.StringType(), False),
            T.StructField("file_name", T.StringType(), False),
            T.StructField("file_size", T.LongType(), True),
            T.StructField("file_modified_time", T.TimestampType(), True),
        ]
    )

    df = spark.createDataFrame(file_rows, schema=schema)

    # Stable fingerprint from metadata (file version identity)
    # Note: cast mtime to string in a consistent format to avoid subtle timestamp formatting diffs.
    df = df.withColumn(
        "file_fingerprint",
        F.sha2(
            F.concat_ws(
                "||",
                F.col("source_system"),
                F.col("sftp_path"),
                F.col("file_name"),
                F.coalesce(F.col("file_size").cast("string"), F.lit("")),
                F.coalesce(
                    F.date_format(
                        F.col("file_modified_time"), "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"
                    ),
                    F.lit(""),
                ),
            ),
            256,
        ),
    )

    return df


def upsert_new_to_manifest(
    spark: pyspark.sql.SparkSession, df_listing: pyspark.sql.DataFrame
) -> None:
    """
    Insert NEW rows for unseen fingerprints only.

    Args:
        spark: Spark session
        df_listing: DataFrame with file listing (must have file_fingerprint column)
    """
    df_manifest_insert = (
        df_listing.select(
            "file_fingerprint",
            "source_system",
            "sftp_path",
            "file_name",
            "file_size",
            "file_modified_time",
        )
        .withColumn("ingested_at", F.lit(None).cast("timestamp"))
        .withColumn("processed_at", F.lit(None).cast("timestamp"))
        .withColumn("status", F.lit("NEW"))
        .withColumn("error_message", F.lit(None).cast("string"))
    )

    df_manifest_insert.createOrReplaceTempView("incoming_manifest_rows")

    spark.sql(
        f"""
        MERGE INTO {MANIFEST_TABLE_PATH} AS t
        USING incoming_manifest_rows AS s
        ON t.file_fingerprint = s.file_fingerprint
        WHEN NOT MATCHED THEN INSERT *
        """
    )


def get_files_to_queue(
    spark: pyspark.sql.SparkSession, df_listing: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Return files that should be queued for downstream processing.

    Criteria:
      - present in current SFTP listing (df_listing)
      - exist in manifest with status = 'NEW'
      - NOT already present in pending_ingest_queue

    Args:
        spark: Spark session
        df_listing: DataFrame with file listing (must have file_fingerprint column)

    Returns:
        DataFrame of files to queue
    """
    manifest_new = (
        spark.table(MANIFEST_TABLE_PATH)
        .select("file_fingerprint", "status")
        .where(F.col("status") == F.lit("NEW"))
        .select("file_fingerprint")
    )

    already_queued = spark.table(QUEUE_TABLE_PATH).select("file_fingerprint").distinct()

    # Only queue files that are:
    #   in current listing AND in manifest NEW AND not in queue
    to_queue = df_listing.join(manifest_new, on="file_fingerprint", how="inner").join(
        already_queued, on="file_fingerprint", how="left_anti"
    )
    return to_queue


def download_new_files_and_queue(
    spark: pyspark.sql.SparkSession,
    sftp: "paramiko.SFTPClient",
    df_new: pyspark.sql.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Download each new file to /tmp and upsert into pending_ingest_queue.

    Args:
        spark: Spark session
        sftp: SFTP client connection
        df_new: DataFrame of files to download and queue
        logger: Optional logger instance (defaults to module logger)

    Returns:
        Number of files queued
    """
    if logger is None:
        logger = LOGGER

    os.makedirs(SFTP_TMP_DIR, exist_ok=True)

    rows = df_new.select(
        "file_fingerprint",
        "source_system",
        "sftp_path",
        "file_name",
        "file_size",
        "file_modified_time",
    ).collect()

    queued = []
    for r in rows:
        fp = r["file_fingerprint"]
        sftp_path = r["sftp_path"]
        file_name = r["file_name"]

        remote_path = f"{sftp_path.rstrip('/')}/{file_name}"
        local_path = os.path.abspath(os.path.join(SFTP_TMP_DIR, f"{fp}__{file_name}"))

        # If local already exists (e.g., rerun), skip re-download
        if not os.path.exists(local_path):
            logger.info(
                f"Downloading new file from SFTP: {remote_path} -> {local_path}"
            )
            download_sftp_atomic(
                sftp, remote_path, local_path, chunk=SFTP_DOWNLOAD_CHUNK_MB
            )
        else:
            logger.info(f"Local file already staged, skipping download: {local_path}")

        queued.append(
            {
                "file_fingerprint": fp,
                "source_system": r["source_system"],
                "sftp_path": sftp_path,
                "file_name": file_name,
                "file_size": r["file_size"],
                "file_modified_time": r["file_modified_time"],
                "local_tmp_path": local_path,
                "queued_at": datetime.now(timezone.utc),
            }
        )

    if not queued:
        return 0

    qschema = T.StructType(
        [
            T.StructField("file_fingerprint", T.StringType(), False),
            T.StructField("source_system", T.StringType(), False),
            T.StructField("sftp_path", T.StringType(), False),
            T.StructField("file_name", T.StringType(), False),
            T.StructField("file_size", T.LongType(), True),
            T.StructField("file_modified_time", T.TimestampType(), True),
            T.StructField("local_tmp_path", T.StringType(), False),
            T.StructField("queued_at", T.TimestampType(), False),
        ]
    )

    df_queue = spark.createDataFrame(queued, schema=qschema)
    df_queue.createOrReplaceTempView("incoming_queue_rows")

    # Upsert into queue (idempotent by fingerprint)
    spark.sql(
        f"""
        MERGE INTO {QUEUE_TABLE_PATH} AS t
        USING incoming_queue_rows AS s
        ON t.file_fingerprint = s.file_fingerprint
        WHEN MATCHED THEN UPDATE SET
        t.local_tmp_path = s.local_tmp_path,
        t.queued_at = s.queued_at
        WHEN NOT MATCHED THEN INSERT *
        """
    )

    return len(queued)


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


def process_and_save_file(volume_dir: str, file_name: str, df: pd.DataFrame) -> str:
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
