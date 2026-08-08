"""
Expand staged queue files into per-institution rows in institution_ingest_plan.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
# Layout: <git_root>/src/edvise/ingestion/nsc_sftp/scripts/<this_file>
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    if _argv0.endswith(".py") and os.path.isfile(_argv0):
        _script_dir = os.path.dirname(_argv0)
    else:
        _script_dir = os.path.abspath(os.getcwd())
_current = _script_dir
for _ in range(8):
    if os.path.isdir(os.path.join(_current, "edvise")):
        if _current not in sys.path:
            sys.path.insert(0, _current)
        break
    _parent = os.path.dirname(_current)
    if _parent == _current:
        break
    _current = _parent

from edvise.ingestion.nsc_sftp import runtime

runtime.bootstrap_catalog()

from pyspark.sql import functions as F
from pyspark.sql import types as T

from edvise.ingestion.nsc_sftp.constants import (
    COLUMN_RENAMES,
    INSTITUTION_COLUMN_PATTERN,
    PLAN_TABLE_PATH,
    QUEUE_TABLE_PATH,
)
from edvise.ingestion.nsc_sftp.helpers import ensure_plan_table, extract_institution_ids

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)
INST_COL_PATTERN = re.compile(INSTITUTION_COLUMN_PATTERN, re.IGNORECASE)

ensure_plan_table(spark, PLAN_TABLE_PATH)
if not spark.catalog.tableExists(QUEUE_TABLE_PATH):
    runtime.notebook_exit(dbutils, "NO_QUEUE_TABLE")

queue_df = spark.table(QUEUE_TABLE_PATH)
if queue_df.limit(1).count() == 0:
    runtime.notebook_exit(dbutils, "NO_QUEUED_FILES")

queue_df = queue_df.join(
    spark.table(PLAN_TABLE_PATH).select("file_fingerprint").distinct(),
    on="file_fingerprint",
    how="left_anti",
)
if queue_df.limit(1).count() == 0:
    runtime.notebook_exit(dbutils, "NO_NEW_EXPANSION_WORK")

queued_files = queue_df.select(
    "file_fingerprint",
    "file_name",
    F.col("local_tmp_path").alias("local_path"),
    "file_size",
    "file_modified_time",
).collect()

work_items: list[dict] = []
missing: list[str] = []
now_ts = datetime.now(timezone.utc)

for row in queued_files:
    fp, file_name, local_path = (
        row["file_fingerprint"],
        row["file_name"],
        row["local_path"],
    )
    if not local_path or not os.path.exists(local_path):
        missing.append(f"fp={fp} file={file_name} path={local_path}")
        continue

    inst_col, inst_ids = extract_institution_ids(
        local_path, renames=COLUMN_RENAMES, inst_col_pattern=INST_COL_PATTERN
    )
    if not inst_col or not inst_ids:
        logger.warning("No institution IDs for file=%s fp=%s; skipping.", file_name, fp)
        continue

    work_items.extend(
        {
            "file_fingerprint": fp,
            "file_name": file_name,
            "local_path": local_path,
            "institution_id": inst_id,
            "inst_col": inst_col,
            "file_size": row["file_size"],
            "file_modified_time": row["file_modified_time"],
            "planned_at": now_ts,
        }
        for inst_id in inst_ids
    )
    logger.info(
        "file=%s: %s institution(s) via %s preview=%s",
        file_name,
        len(inst_ids),
        inst_col,
        inst_ids[:10],
    )

if missing:
    raise FileNotFoundError("Missing staged files: " + "; ".join(missing))
if not work_items:
    runtime.notebook_exit(dbutils, "NO_WORK_ITEMS")

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
spark.createDataFrame(work_items, schema=schema).createOrReplaceTempView(
    "incoming_plan_rows"
)
spark.sql(
    f"""
    MERGE INTO {PLAN_TABLE_PATH} AS t
    USING incoming_plan_rows AS s
    ON t.file_fingerprint = s.file_fingerprint AND t.institution_id = s.institution_id
    WHEN MATCHED THEN UPDATE SET
      t.file_name = s.file_name,
      t.local_path = s.local_path,
      t.inst_col = s.inst_col,
      t.file_size = s.file_size,
      t.file_modified_time = s.file_modified_time,
      t.planned_at = s.planned_at
    WHEN NOT MATCHED THEN INSERT *
    """
)
logger.info("Wrote/updated %s plan row(s) into %s", len(work_items), PLAN_TABLE_PATH)
runtime.notebook_exit(dbutils, f"WORK_ITEMS={len(work_items)}")
