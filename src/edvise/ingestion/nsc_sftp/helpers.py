"""
NSC SFTP ingestion helpers (manifest, queue, plan, staging, bronze writes).

NSC-specific utilities for processing SFTP files, extracting institution IDs,
managing ingestion manifests, and working with Databricks schemas/volumes.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    import paramiko
    from edvise.utils.api_requests import EdviseAPIClient

import pandas as pd
import pyspark.sql
from pyspark.sql import functions as F
from pyspark.sql import types as T

from edvise.ingestion.nsc_sftp.constants import (
    CATALOG,
    DEFAULT_SCHEMA,
    MANIFEST_TABLE_PATH,
    PLAN_TABLE_PATH,
    QUEUE_TABLE_PATH,
    SFTP_DOWNLOAD_CHUNK_MB,
    SFTP_TMP_DIR,
    SFTP_TMP_VOLUME_FQN,
    SFTP_TMP_VOLUME_NAME,
    SFTP_VERIFY_DOWNLOAD,
)
from edvise.utils.data_cleaning import convert_to_snake_case, detect_institution_column
from edvise.utils.sftp import download_sftp_atomic

LOGGER = logging.getLogger(__name__)


def _ensure_sftp_staging_volume_exists(spark: pyspark.sql.SparkSession) -> None:
    """
    Ensure the configured UC volume used for SFTP staging exists and is accessible.

    We stage files to a Unity Catalog volume (CATALOG.default.tmp) so paths remain
    valid across workflow tasks/clusters.
    """
    try:
        rows = spark.sql(f"SHOW VOLUMES IN {CATALOG}.{DEFAULT_SCHEMA}").collect()
    except Exception as e:
        raise RuntimeError(
            f"Failed to verify staging volume exists. Expected UC volume: {SFTP_TMP_VOLUME_FQN}. "
            f"Could not list volumes in {CATALOG}.{DEFAULT_SCHEMA}: {e}"
        ) from e

    def _volume_name(row: pyspark.sql.Row) -> str:
        d = row.asDict()
        for k in ["volume_name", "volumeName", "name"]:
            v = d.get(k)
            if v:
                return str(v)
        return str(list(d.values())[0])

    volume_names = {_volume_name(r) for r in rows}
    if SFTP_TMP_VOLUME_NAME not in volume_names:
        raise RuntimeError(
            f"Required staging UC volume not found: {SFTP_TMP_VOLUME_FQN}. "
            "Create it before running NSC ingestion."
        )

    if not os.path.isdir(SFTP_TMP_DIR):
        raise RuntimeError(
            f"UC volume exists but filesystem path is not accessible: {SFTP_TMP_DIR}. "
            f"Expected UC volume: {SFTP_TMP_VOLUME_FQN}."
        )


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
          run_id STRING,
          cohort ARRAY<STRING>,
          cohort_term_pairs ARRAY<STRUCT<cohort: STRING, cohort_term: STRING>>,
          student_count BIGINT,
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


def reset_files_for_reingest(
    spark: pyspark.sql.SparkSession, file_fingerprints: Sequence[str]
) -> int:
    """
    Clear queue/plan rows and reset manifest to NEW for the given fingerprints.

    Used when ``force_reingest=true`` so previously BRONZE_WRITTEN files can be
    staged and processed again.
    """
    fps = sorted({str(fp).strip() for fp in file_fingerprints if str(fp).strip()})
    if not fps:
        return 0

    spark.createDataFrame(
        [(fp,) for fp in fps],
        schema=T.StructType([T.StructField("file_fingerprint", T.StringType(), False)]),
    ).createOrReplaceTempView("_nsc_reingest_fps")

    if spark.catalog.tableExists(QUEUE_TABLE_PATH):
        spark.sql(
            f"""
            DELETE FROM {QUEUE_TABLE_PATH}
            WHERE file_fingerprint IN (SELECT file_fingerprint FROM _nsc_reingest_fps)
            """
        )
    if spark.catalog.tableExists(PLAN_TABLE_PATH):
        spark.sql(
            f"""
            DELETE FROM {PLAN_TABLE_PATH}
            WHERE file_fingerprint IN (SELECT file_fingerprint FROM _nsc_reingest_fps)
            """
        )
    if spark.catalog.tableExists(MANIFEST_TABLE_PATH):
        cols = set(spark.table(MANIFEST_TABLE_PATH).columns)
        set_parts = ["status = 'NEW'"]
        for col, expr in (
            ("error_message", "NULL"),
            ("ingested_at", "NULL"),
            ("processed_at", "NULL"),
            ("run_id", "NULL"),
        ):
            if col in cols:
                set_parts.append(f"{col} = {expr}")
        spark.sql(
            f"""
            UPDATE {MANIFEST_TABLE_PATH}
            SET {", ".join(set_parts)}
            WHERE file_fingerprint IN (SELECT file_fingerprint FROM _nsc_reingest_fps)
            """
        )
    LOGGER.info("force_reingest reset %s fingerprint(s)", len(fps))
    return len(fps)


def upsert_new_to_manifest(
    spark: pyspark.sql.SparkSession, df_listing: pyspark.sql.DataFrame
) -> None:
    """
    Insert NEW rows for unseen fingerprints only.

    Args:
        spark: Spark session
        df_listing: DataFrame with file listing (must have file_fingerprint column)
    """
    target_cols = set(spark.table(MANIFEST_TABLE_PATH).columns)

    df_manifest_insert = df_listing.select(
        "file_fingerprint",
        "source_system",
        "sftp_path",
        "file_name",
        "file_size",
        "file_modified_time",
    )

    # Only set optional columns if they exist on the target table. We avoid ALTER TABLE
    # here by staying backward-compatible with older manifest schemas.
    if "run_id" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "run_id", F.lit(None).cast("string")
        )
    if "cohort" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "cohort", F.lit(None).cast("array<string>")
        )
    if "cohort_term_pairs" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "cohort_term_pairs",
            F.lit(None).cast("array<struct<cohort:string,cohort_term:string>>"),
        )
    if "student_count" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "student_count", F.lit(None).cast("bigint")
        )

    if "ingested_at" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "ingested_at", F.lit(None).cast("timestamp")
        )
    if "processed_at" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "processed_at", F.lit(None).cast("timestamp")
        )
    if "status" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn("status", F.lit("NEW"))
    if "error_message" in target_cols:
        df_manifest_insert = df_manifest_insert.withColumn(
            "error_message", F.lit(None).cast("string")
        )

    df_manifest_insert.createOrReplaceTempView("incoming_manifest_rows")

    cols = df_manifest_insert.columns
    cols_sql = ", ".join([f"`{c}`" for c in cols])
    vals_sql = ", ".join([f"s.`{c}`" for c in cols])

    spark.sql(
        f"""
        MERGE INTO {MANIFEST_TABLE_PATH} AS t
        USING incoming_manifest_rows AS s
        ON t.file_fingerprint = s.file_fingerprint
        WHEN NOT MATCHED THEN INSERT ({cols_sql}) VALUES ({vals_sql})
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
    sftp: paramiko.SFTPClient,
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
    _ensure_sftp_staging_volume_exists(spark)

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
                sftp,
                remote_path,
                local_path,
                chunk=SFTP_DOWNLOAD_CHUNK_MB,
                verify=SFTP_VERIFY_DOWNLOAD,
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


_PLAN_IDENTITY_COLS = ("inst_id", "institution_name")


def ensure_plan_table(spark: pyspark.sql.SparkSession, plan_table: str) -> None:
    """
    Create institution_ingest_plan if missing; add SST identity columns if needed.

    ``institution_id`` = PDP id from file; ``inst_id`` / ``institution_name`` = SST API.
    """
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {plan_table} (
          file_fingerprint STRING,
          file_name STRING,
          local_path STRING,
          institution_id STRING,
          inst_id STRING,
          institution_name STRING,
          inst_col STRING,
          file_size BIGINT,
          file_modified_time TIMESTAMP,
          planned_at TIMESTAMP
        )
        USING DELTA
        """
    )
    existing = {f.name for f in spark.table(plan_table).schema.fields}
    missing = [c for c in _PLAN_IDENTITY_COLS if c not in existing]
    if missing:
        spark.sql(
            f"ALTER TABLE {plan_table} ADD COLUMNS ({', '.join(f'{c} STRING' for c in missing)})"
        )


def resolve_sst_institution(
    api_client: "EdviseAPIClient", pdp_id: str
) -> tuple[str, str]:
    """Return ``(inst_id, institution_name)`` from SST institutions API."""
    from edvise.utils.api_requests import fetch_institution_by_pdp_id

    info = fetch_institution_by_pdp_id(api_client, pdp_id)
    inst_id = str(info.get("inst_id") or "").strip()
    name = str(info.get("name") or "").strip()
    if not inst_id or not name:
        raise ValueError(
            f"SST institution response missing inst_id/name for pdp_id={pdp_id}: "
            f"keys={list(info.keys())}"
        )
    return inst_id, name


def sst_identity_or_resolve(
    api_client: "EdviseAPIClient",
    pdp_id: str,
    planned: Optional[tuple[str, str]] = None,
) -> tuple[str, str]:
    """Use planned ``(inst_id, name)`` when present; otherwise call SST API."""
    if planned and planned[0] and planned[1]:
        return planned
    return resolve_sst_institution(api_client, pdp_id)


def group_plan_rows_by_file(
    plan_rows: Sequence[Any],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, list[str]],
    dict[str, dict[str, tuple[str, str]]],
]:
    """
    Group collected plan Rows into per-file metadata, PDP ids, and SST identities.

    Returns ``(by_file, inst_ids_by_fp, identity_by_fp)``.
    """
    by_file: dict[str, dict[str, str]] = {}
    inst_ids_by_fp: dict[str, list[str]] = {}
    identity_by_fp: dict[str, dict[str, tuple[str, str]]] = {}
    for row in plan_rows:
        raw: Mapping[str, Any]
        if hasattr(row, "asDict"):
            raw = row.asDict()
        else:
            raw = row
        fp = str(raw["file_fingerprint"])
        pdp_id = str(raw["institution_id"])
        inst_ids_by_fp.setdefault(fp, []).append(pdp_id)
        by_file.setdefault(
            fp,
            {
                "file_name": str(raw["file_name"]),
                "local_path": str(raw["local_path"]),
                "inst_col": str(raw["inst_col"]),
            },
        )
        sst_id = str(raw.get("inst_id") or "").strip()
        sst_name = str(raw.get("institution_name") or "").strip()
        if sst_id and sst_name:
            identity_by_fp.setdefault(fp, {})[pdp_id] = (sst_id, sst_name)
    return by_file, inst_ids_by_fp, identity_by_fp


def resolve_sst_institutions(
    api_client: "EdviseAPIClient", pdp_ids: Sequence[str]
) -> dict[str, tuple[str, str]]:
    """
    Resolve distinct PDP ids → ``(inst_id, institution_name)``.

    Soft-skips ids the SST API cannot resolve (e.g. 404); logs a warning per miss.
    """
    resolved: dict[str, tuple[str, str]] = {}
    for pdp_id in sorted({str(p).strip() for p in pdp_ids if str(p).strip()}):
        try:
            resolved[pdp_id] = resolve_sst_institution(api_client, pdp_id)
        except Exception as exc:
            LOGGER.warning("SST unresolved pdp_id=%s: %s", pdp_id, exc)
    return resolved


def log_labeled_lines(logger: logging.Logger, label: str, lines: Sequence[str]) -> None:
    """Emit a labeled end-of-task summary block (one line per item)."""
    logger.info("=== %s (%s) ===", label, len(lines))
    if not lines:
        logger.info("%s (none)", label)
        return
    for line in lines:
        logger.info("%s %s", label, line)


def backfill_plan_institution_identity(
    spark: pyspark.sql.SparkSession,
    api_client: "EdviseAPIClient",
    plan_table: str,
) -> int:
    """Fill blank plan ``inst_id`` / ``institution_name`` via SST API. Soft-fails on auth."""
    ensure_plan_table(spark, plan_table)
    blank = F.coalesce(F.length(F.trim(F.col("inst_id"))), F.lit(0)) == 0
    blank = blank | (
        F.coalesce(F.length(F.trim(F.col("institution_name"))), F.lit(0)) == 0
    )
    pdp_ids = [
        str(r["institution_id"]).strip()
        for r in spark.table(plan_table)
        .where(F.col("institution_id").isNotNull() & blank)
        .select("institution_id")
        .distinct()
        .collect()
        if r["institution_id"]
    ]
    if not pdp_ids:
        return 0
    try:
        resolved = resolve_sst_institutions(api_client, pdp_ids)
    except Exception:
        LOGGER.exception(
            "SST backfill failed for %s PDP id(s); leaving inst_id/name blank",
            len(pdp_ids),
        )
        return 0
    rows = [
        {
            "institution_id": pdp_id,
            "inst_id": inst_id,
            "institution_name": name,
        }
        for pdp_id, (inst_id, name) in resolved.items()
    ]
    spark.createDataFrame(rows).createOrReplaceTempView("plan_identity_backfill")
    spark.sql(
        f"""
        MERGE INTO {plan_table} AS t
        USING plan_identity_backfill AS s
        ON t.institution_id = s.institution_id
        WHEN MATCHED THEN UPDATE SET
          t.inst_id = s.inst_id,
          t.institution_name = s.institution_name
        """
    )
    LOGGER.info("Backfilled SST identity for %s PDP id(s)", len(rows))
    return len(rows)


_PLAN_ROW_SCHEMA = T.StructType(
    [
        T.StructField("file_fingerprint", T.StringType(), False),
        T.StructField("file_name", T.StringType(), False),
        T.StructField("local_path", T.StringType(), False),
        T.StructField("institution_id", T.StringType(), False),
        T.StructField("inst_id", T.StringType(), True),
        T.StructField("institution_name", T.StringType(), True),
        T.StructField("inst_col", T.StringType(), False),
        T.StructField("file_size", T.LongType(), True),
        T.StructField("file_modified_time", T.TimestampType(), True),
        T.StructField("planned_at", T.TimestampType(), False),
    ]
)


def merge_institution_plan_rows(
    spark: pyspark.sql.SparkSession,
    plan_table: str,
    work_items: list[dict],
) -> int:
    """MERGE plan rows keyed by ``(file_fingerprint, institution_id)``."""
    if not work_items:
        return 0
    ensure_plan_table(spark, plan_table)
    spark.createDataFrame(work_items, schema=_PLAN_ROW_SCHEMA).createOrReplaceTempView(
        "incoming_plan_rows"
    )
    spark.sql(
        f"""
        MERGE INTO {plan_table} AS t
        USING incoming_plan_rows AS s
        ON t.file_fingerprint = s.file_fingerprint AND t.institution_id = s.institution_id
        WHEN MATCHED THEN UPDATE SET
          t.file_name = s.file_name,
          t.local_path = s.local_path,
          t.inst_id = s.inst_id,
          t.institution_name = s.institution_name,
          t.inst_col = s.inst_col,
          t.file_size = s.file_size,
          t.file_modified_time = s.file_modified_time,
          t.planned_at = s.planned_at
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    return len(work_items)


def _normalize_header_map(
    header_cols: list[str], renames: dict[str, str]
) -> dict[str, str]:
    """Map raw CSV header -> normalized/renamed column name."""
    out: dict[str, str] = {}
    for raw in header_cols:
        normalized = convert_to_snake_case(raw)
        out[raw] = renames.get(normalized, normalized)
    return out


def normalize_staged_frame(
    df: pd.DataFrame, *, renames: dict[str, str]
) -> pd.DataFrame:
    """Apply snake_case + COLUMN_RENAMES to a staged PDP frame."""
    return df.rename(columns=_normalize_header_map(list(df.columns), renames))


def load_staged_csv(
    local_path: str,
    *,
    renames: dict[str, str],
    inst_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a staged CSV once with institution IDs forced to string when possible.
    """
    header_cols = pd.read_csv(local_path, nrows=0).columns.tolist()
    header_map = _normalize_header_map(header_cols, renames)
    dtype = None
    if inst_col:
        raw_inst_col = next(
            (raw for raw, norm in header_map.items() if norm == inst_col), None
        )
        if raw_inst_col:
            dtype = {raw_inst_col: str}
    df = pd.read_csv(local_path, on_bad_lines="warn", dtype=dtype)
    return normalize_staged_frame(df, renames=renames)


def summarize_file_metrics(
    df: pd.DataFrame,
) -> tuple[Optional[int], Optional[list[str]], Optional[list[dict[str, str]]]]:
    """Cheap file-level metrics for manifest updates / logging."""
    student_count = None
    student_col = next(
        (c for c in ("student_id", "study_id", "student_guid") if c in df.columns),
        None,
    )
    if student_col:
        student_count = int(df[student_col].nunique(dropna=True))

    file_cohort = None
    if "cohort" in df.columns:
        vals = df["cohort"].dropna().astype(str).str.strip()
        vals = vals[~vals.str.lower().isin({"", "nan", "none", "null"})]
        uniq = sorted(vals.unique().tolist())
        file_cohort = uniq or None

    file_cohort_term_pairs = None
    if {"cohort", "cohort_term"}.issubset(df.columns):
        tmp = df.loc[:, ["cohort", "cohort_term"]].dropna().copy()
        tmp["cohort"] = tmp["cohort"].astype(str).str.strip()
        tmp["cohort_term"] = tmp["cohort_term"].astype(str).str.strip().str.upper()
        bad = {"", "nan", "none", "null"}
        tmp = tmp[
            ~tmp["cohort"].str.lower().isin(bad)
            & ~tmp["cohort_term"].str.lower().isin(bad)
        ]
        tmp = tmp.drop_duplicates().sort_values(["cohort", "cohort_term"])
        pairs = [
            {"cohort": r.cohort, "cohort_term": r.cohort_term}
            for r in tmp.itertuples(index=False)
        ]
        file_cohort_term_pairs = pairs or None

    return student_count, file_cohort, file_cohort_term_pairs


def _normalize_institution_id(value: object) -> Optional[str]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return str(int(value)) if value.is_integer() else str(value).strip()
    except Exception:
        pass

    s = str(value).strip()
    if s == "" or s.lower() in {
        "nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        return None
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]
    return s


def extract_institution_ids(
    local_path: str,
    *,
    renames: dict[str, str],
    inst_col_pattern: re.Pattern,
) -> tuple[Optional[str], list[str]]:
    """
    Extract unique institution IDs from a staged CSV.

    Only the institution column is fully read (header scan first), which keeps
    stage-02 cheap for wide PDP files.
    """
    header_cols = pd.read_csv(local_path, nrows=0).columns.tolist()
    header_map = _normalize_header_map(header_cols, renames)
    inst_col = detect_institution_column(list(header_map.values()), inst_col_pattern)
    if inst_col is None:
        return None, []

    raw_inst_col = next(raw for raw, norm in header_map.items() if norm == inst_col)
    series = pd.read_csv(local_path, usecols=[raw_inst_col], on_bad_lines="warn")[
        raw_inst_col
    ].dropna()

    ids: set[str] = set()
    for value in series.tolist():
        normalized = _normalize_institution_id(value)
        if normalized is not None:
            ids.add(normalized)
    return inst_col, sorted(ids)


def resolve_bronze_volume_dir(
    spark: pyspark.sql.SparkSession, catalog: str, inst_prefix: str
) -> str:
    """Return `/Volumes/{catalog}/{schema}/{volume}` for an institution prefix."""
    from edvise.utils.databricks import find_bronze_schema, find_bronze_volume_name

    bronze_schema = find_bronze_schema(spark, catalog, inst_prefix)
    bronze_volume_name = find_bronze_volume_name(spark, catalog, bronze_schema)
    return f"/Volumes/{catalog}/{bronze_schema}/{bronze_volume_name}"


def bronze_written_file_names(spark: pyspark.sql.SparkSession) -> set[str]:
    """file_name values already marked BRONZE_WRITTEN in ingestion_manifest."""
    if not spark.catalog.tableExists(MANIFEST_TABLE_PATH):
        return set()
    rows = (
        spark.table(MANIFEST_TABLE_PATH)
        .where(F.col("status") == F.lit("BRONZE_WRITTEN"))
        .select("file_name")
        .collect()
    )
    return {r["file_name"] for r in rows if r["file_name"]}


def build_edvise_api_client(
    *,
    api_key: str,
    db_workspace: str,
    token_path: str,
    institution_lookup_path: str,
) -> "EdviseAPIClient":
    """Construct EdviseAPIClient with workspace-derived base URL."""
    from edvise.utils.api_requests import EdviseAPIClient, get_base_url

    return EdviseAPIClient(
        api_key=api_key,
        base_url=get_base_url(db_workspace),
        token_endpoint=token_path,
        institution_lookup_path=institution_lookup_path,
    )


def job_edvise_api_client(
    dbutils_obj: object,
    *,
    db_workspace: str,
    secret_scope: str,
    sst_api_key_secret_key: str,
    token_path: Optional[str] = None,
    institution_lookup_path: Optional[str] = None,
) -> "EdviseAPIClient":
    """Build API client from Databricks secret scope/key names (job params)."""
    from edvise.ingestion.nsc_sftp.constants import (
        INSTITUTION_LOOKUP_PATH,
        SST_TOKEN_PATH,
    )

    api_key = dbutils_obj.secrets.get(  # type: ignore[attr-defined]
        scope=secret_scope, key=sst_api_key_secret_key
    ).strip()
    if not api_key:
        raise RuntimeError(
            f"Empty SST API key: scope={secret_scope} key={sst_api_key_secret_key}"
        )
    return build_edvise_api_client(
        api_key=api_key,
        db_workspace=db_workspace,
        token_path=token_path or SST_TOKEN_PATH,
        institution_lookup_path=institution_lookup_path or INSTITUTION_LOOKUP_PATH,
    )


def update_manifest(
    spark: pyspark.sql.SparkSession,
    manifest_table: str,
    file_fingerprint: str,
    *,
    status: str,
    error_message: Optional[str],
    run_id: Optional[str] = None,
    cohort: Optional[list[str]] = None,
    cohort_term_pairs: Optional[list[dict[str, str]]] = None,
    student_count: Optional[int] = None,
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

    target_cols = set(spark.table(manifest_table).columns)

    # ingested_at only set when we finish BRONZE_WRITTEN
    row: dict[str, object] = {
        "file_fingerprint": file_fingerprint,
        "status": status,
        "error_message": error_message,
    }

    fields = [
        T.StructField("file_fingerprint", T.StringType(), False),
        T.StructField("status", T.StringType(), False),
        T.StructField("error_message", T.StringType(), True),
    ]
    if "run_id" in target_cols:
        row["run_id"] = run_id
        fields.append(T.StructField("run_id", T.StringType(), True))
    if "cohort" in target_cols:
        row["cohort"] = cohort
        fields.append(T.StructField("cohort", T.ArrayType(T.StringType()), True))
    if "cohort_term_pairs" in target_cols:
        row["cohort_term_pairs"] = cohort_term_pairs
        fields.append(
            T.StructField(
                "cohort_term_pairs",
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("cohort", T.StringType(), True),
                            T.StructField("cohort_term", T.StringType(), True),
                        ]
                    )
                ),
                True,
            )
        )
    if "student_count" in target_cols:
        row["student_count"] = student_count
        fields.append(T.StructField("student_count", T.LongType(), True))
    if "ingested_at" in target_cols:
        row["ingested_at"] = now_ts if status == "BRONZE_WRITTEN" else None
        fields.append(T.StructField("ingested_at", T.TimestampType(), True))
    if "processed_at" in target_cols:
        row["processed_at"] = now_ts
        fields.append(T.StructField("processed_at", T.TimestampType(), False))

    schema = T.StructType(fields)
    df = spark.createDataFrame([row], schema=schema)
    df.createOrReplaceTempView("manifest_updates")

    set_clauses = [
        "t.status = s.status",
        "t.error_message = s.error_message",
    ]
    if "run_id" in target_cols:
        set_clauses.append("t.run_id = COALESCE(s.run_id, t.run_id)")
    if "cohort" in target_cols:
        set_clauses.append("t.cohort = COALESCE(s.cohort, t.cohort)")
    if "cohort_term_pairs" in target_cols:
        set_clauses.append(
            "t.cohort_term_pairs = COALESCE(s.cohort_term_pairs, t.cohort_term_pairs)"
        )
    if "student_count" in target_cols:
        set_clauses.append(
            "t.student_count = COALESCE(s.student_count, t.student_count)"
        )
    if "ingested_at" in target_cols:
        set_clauses.append("t.ingested_at = COALESCE(s.ingested_at, t.ingested_at)")
    if "processed_at" in target_cols:
        set_clauses.append("t.processed_at = s.processed_at")

    set_sql = ",\n          ".join(set_clauses)
    spark.sql(
        f"""
        MERGE INTO {manifest_table} AS t
        USING manifest_updates AS s
        ON t.file_fingerprint = s.file_fingerprint
        WHEN MATCHED THEN UPDATE SET
          {set_sql}
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
