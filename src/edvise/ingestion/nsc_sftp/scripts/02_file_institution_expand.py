"""
Read staged files from pending_ingest_queue, detect institution ID column,
expand to per-institution rows, and MERGE into institution_ingest_plan.

No SFTP, no API calls, no volume writes beyond reading staged paths.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from edvise.ingestion.constants import (
    COLUMN_RENAMES,
    INSTITUTION_COLUMN_PATTERN,
    PLAN_TABLE_PATH,
    QUEUE_TABLE_PATH,
)
from edvise.ingestion.nsc_sftp_helpers import ensure_plan_table, extract_institution_ids

try:
    dbutils  # noqa: F821
except NameError:
    from unittest.mock import MagicMock

    dbutils = MagicMock()

spark = DatabricksSession.builder.getOrCreate()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INST_COL_PATTERN = re.compile(INSTITUTION_COLUMN_PATTERN, re.IGNORECASE)

ensure_plan_table(spark, PLAN_TABLE_PATH)

if not spark.catalog.tableExists(QUEUE_TABLE_PATH):
    logger.info(f"Queue table {QUEUE_TABLE_PATH} not found. Exiting (no-op).")
    dbutils.notebook.exit("NO_QUEUE_TABLE")

queue_df = spark.read.table(QUEUE_TABLE_PATH)

if queue_df.limit(1).count() == 0:
    logger.info("pending_ingest_queue is empty. Exiting (no-op).")
    dbutils.notebook.exit("NO_QUEUED_FILES")

existing_fp = (
    spark.table(PLAN_TABLE_PATH).select("file_fingerprint").distinct()
    if spark.catalog.tableExists(PLAN_TABLE_PATH)
    else None
)
if existing_fp is not None:
    queue_df = queue_df.join(existing_fp, on="file_fingerprint", how="left_anti")

if queue_df.limit(1).count() == 0:
    logger.info(
        "All queued files have already been expanded into institution work items. Exiting (no-op)."
    )
    dbutils.notebook.exit("NO_NEW_EXPANSION_WORK")

logger.info("Queued files to expand preview (after excluding already-expanded):")
queue_df.select("file_fingerprint", "file_name", "local_tmp_path", "queued_at").show(
    25, truncate=False
)

queued_files = queue_df.select(
    "file_fingerprint",
    "file_name",
    F.col("local_tmp_path").alias("local_path"),
    "file_size",
    "file_modified_time",
).collect()

logger.info(
    f"Expanding {len(queued_files)} staged file(s) into per-institution work items..."
)

work_items = []
missing_files = []

for r in queued_files:
    fp = r["file_fingerprint"]
    file_name = r["file_name"]
    local_path = r["local_path"]

    if not local_path or not os.path.exists(local_path):
        missing_files.append((fp, file_name, local_path))
        continue

    try:
        inst_col, inst_ids = extract_institution_ids(
            local_path, renames=COLUMN_RENAMES, inst_col_pattern=INST_COL_PATTERN
        )
        if inst_col is None:
            logger.warning(
                f"No institution id column found for file={file_name} fp={fp}. Skipping this file."
            )
            continue

        if not inst_ids:
            logger.warning(
                f"Institution column found but no IDs present for file={file_name} fp={fp}. Skipping."
            )
            continue

        now_ts = datetime.now(timezone.utc)
        for inst_id in inst_ids:
            work_items.append(
                {
                    "file_fingerprint": fp,
                    "file_name": file_name,
                    "local_path": local_path,
                    "institution_id": inst_id,
                    "inst_col": inst_col,
                    "file_size": r["file_size"],
                    "file_modified_time": r["file_modified_time"],
                    "planned_at": now_ts,
                }
            )

        preview_ids = inst_ids[:10]
        logger.info(
            f"file={file_name} fp={fp}: found {len(inst_ids)} institution id(s) using column '{inst_col}'. "
            f"Preview first 10 IDs={preview_ids}"
        )

    except Exception as e:
        logger.exception(f"Failed expanding file={file_name} fp={fp}: {e}")
        raise

if missing_files:
    msg = (
        "Some staged files are missing on disk (staging path missing/inaccessible). "
        + "; ".join([f"fp={fp} file={fn} path={lp}" for fp, fn, lp in missing_files])
    )
    logger.error(msg)
    raise FileNotFoundError(msg)

if not work_items:
    logger.info("No work items generated from staged files. Exiting (no-op).")
    dbutils.notebook.exit("NO_WORK_ITEMS")

schema = T.StructType(
    [
        T.StructField("file_fingerprint", T.StringType(), False),
        T.StructField("file_name", T.StringType(), False),
        T.StructField("local_path", T.StringType(), False),
        T.StructField("institution_id", T.StringType(), False),
        T.StructField("inst_col", T.StringType(), False),
        T.StructField("file_size", T.LongType(), True),
        T.StructField("file_modified_time", T.TimestampType(), True),
        T.StructField("planned_at", T.TimestampType(), False),
    ]
)

df_plan = spark.createDataFrame(work_items, schema=schema)

logger.info("Work items summary by file (distinct institutions):")
df_plan.groupBy("file_name").agg(
    F.countDistinct("institution_id").alias("institution_count")
).orderBy("file_name").show(truncate=False)

df_plan.createOrReplaceTempView("incoming_plan_rows")

spark.sql(
    f"""
    MERGE INTO {PLAN_TABLE_PATH} AS t
    USING incoming_plan_rows AS s
    ON  t.file_fingerprint = s.file_fingerprint
    AND t.institution_id   = s.institution_id
    WHEN MATCHED THEN UPDATE SET
      t.file_name          = s.file_name,
      t.local_path         = s.local_path,
      t.inst_col           = s.inst_col,
      t.file_size          = s.file_size,
      t.file_modified_time = s.file_modified_time,
      t.planned_at         = s.planned_at
    WHEN NOT MATCHED THEN INSERT *
    """
)

count_out = df_plan.count()
logger.info(
    f"Wrote/updated {count_out} institution work item(s) into {PLAN_TABLE_PATH}."
)
dbutils.notebook.exit(f"WORK_ITEMS={count_out}")
