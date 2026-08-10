"""
Expand staged queue files into per-institution rows in institution_ingest_plan.

Stores PDP id (from file) plus SST ``inst_id`` / ``name`` from the institutions API.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    _script_dir = (
        os.path.dirname(_argv0)
        if _argv0.endswith(".py") and os.path.isfile(_argv0)
        else os.path.abspath(os.getcwd())
    )
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
    INSTITUTION_LOOKUP_PATH,
    PLAN_TABLE_PATH,
    QUEUE_TABLE_PATH,
    SST_TOKEN_PATH,
)
from edvise.ingestion.nsc_sftp.helpers import (
    backfill_plan_institution_identity,
    ensure_plan_table,
    extract_institution_ids,
    job_edvise_api_client,
    resolve_sst_institutions,
)

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)
INST_COL_PATTERN = re.compile(INSTITUTION_COLUMN_PATTERN, re.IGNORECASE)

api_client = job_edvise_api_client(
    dbutils,
    db_workspace=runtime.require_job_param("DB_workspace"),
    secret_scope=runtime.require_job_param("nsc_sftp_secret_scope"),
    sst_api_key_secret_key=runtime.require_job_param("sst_api_key_secret_key"),
    token_path=SST_TOKEN_PATH,
    institution_lookup_path=INSTITUTION_LOOKUP_PATH,
)

ensure_plan_table(spark, PLAN_TABLE_PATH)
backfilled = backfill_plan_institution_identity(spark, api_client, PLAN_TABLE_PATH)
if backfilled:
    logger.info("Backfilled SST identity for %s PDP id(s)", backfilled)

if not spark.catalog.tableExists(QUEUE_TABLE_PATH):
    runtime.notebook_exit(dbutils, f"NO_QUEUE_TABLE;BACKFILLED={backfilled}")

queue_df = spark.table(QUEUE_TABLE_PATH)
if queue_df.limit(1).count() == 0:
    runtime.notebook_exit(dbutils, f"NO_QUEUED_FILES;BACKFILLED={backfilled}")

queue_df = queue_df.join(
    spark.table(PLAN_TABLE_PATH).select("file_fingerprint").distinct(),
    on="file_fingerprint",
    how="left_anti",
)
if queue_df.limit(1).count() == 0:
    runtime.notebook_exit(dbutils, f"NO_NEW_EXPANSION_WORK;BACKFILLED={backfilled}")

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
file_pdp_ids: dict[str, tuple[object, list[str], str]] = {}

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
    file_pdp_ids[fp] = (row, inst_ids, inst_col)

if missing:
    raise FileNotFoundError("Missing staged files: " + "; ".join(missing))
if not file_pdp_ids:
    runtime.notebook_exit(dbutils, f"NO_WORK_ITEMS;BACKFILLED={backfilled}")

all_pdp_ids = [pid for _, ids, _ in file_pdp_ids.values() for pid in ids]
sst_by_pdp = resolve_sst_institutions(api_client, all_pdp_ids)

for fp, (row, inst_ids, inst_col) in file_pdp_ids.items():
    for pdp_id in inst_ids:
        sst_inst_id, institution_name = sst_by_pdp[pdp_id]
        work_items.append(
            {
                "file_fingerprint": fp,
                "file_name": row["file_name"],
                "local_path": row["local_path"],
                "institution_id": pdp_id,
                "inst_id": sst_inst_id,
                "institution_name": institution_name,
                "inst_col": inst_col,
                "file_size": row["file_size"],
                "file_modified_time": row["file_modified_time"],
                "planned_at": now_ts,
            }
        )
    logger.info(
        "file=%s: %s institution(s) via %s preview=%s",
        row["file_name"],
        len(inst_ids),
        inst_col,
        [
            {"pdp_id": pid, "inst_id": sst_by_pdp[pid][0], "name": sst_by_pdp[pid][1]}
            for pid in inst_ids[:10]
        ],
    )

schema = T.StructType(
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
      t.inst_id = s.inst_id,
      t.institution_name = s.institution_name,
      t.inst_col = s.inst_col,
      t.file_size = s.file_size,
      t.file_modified_time = s.file_modified_time,
      t.planned_at = s.planned_at
    WHEN NOT MATCHED THEN INSERT *
    """
)
logger.info("Wrote/updated %s plan row(s) into %s", len(work_items), PLAN_TABLE_PATH)
runtime.notebook_exit(dbutils, f"WORK_ITEMS={len(work_items)};BACKFILLED={backfilled}")
