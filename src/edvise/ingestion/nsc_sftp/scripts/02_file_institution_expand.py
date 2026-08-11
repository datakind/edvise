"""
Expand staged queue files into per-institution rows in institution_ingest_plan.

Stores PDP id (from file) plus SST ``inst_id`` / ``name`` from the institutions API.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

_cur = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0] or "."))
for _ in range(8):
    if os.path.isdir(os.path.join(_cur, "edvise")):
        if _cur not in sys.path:
            sys.path.insert(0, _cur)
        break
    _nxt = os.path.dirname(_cur)
    if _nxt == _cur:
        break
    _cur = _nxt

from edvise.ingestion.nsc_sftp import runtime  # noqa: E402

runtime.bootstrap_catalog()

from pyspark.sql import functions as F

from edvise.ingestion.nsc_sftp.constants import (
    COLUMN_RENAMES,
    INSTITUTION_COLUMN_PATTERN,
    PLAN_TABLE_PATH,
    QUEUE_TABLE_PATH,
)
from edvise.ingestion.nsc_sftp.helpers import (
    backfill_plan_institution_identity,
    ensure_plan_table,
    extract_institution_ids,
    merge_institution_plan_rows,
    resolve_sst_institutions,
)

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)
INST_COL_PATTERN = re.compile(INSTITUTION_COLUMN_PATTERN, re.IGNORECASE)

api_client = runtime.require_edvise_api_client(dbutils)
ensure_plan_table(spark, PLAN_TABLE_PATH)
backfilled = backfill_plan_institution_identity(spark, api_client, PLAN_TABLE_PATH)

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

pending: dict[str, tuple[dict[str, Any], list[str], str]] = {}
missing: list[str] = []
now_ts = datetime.now(timezone.utc)

for row in queued_files:
    fp = row["file_fingerprint"]
    file_name = row["file_name"]
    local_path = row["local_path"]
    if not local_path or not os.path.exists(local_path):
        missing.append(f"fp={fp} file={file_name} path={local_path}")
        continue
    inst_col, inst_ids = extract_institution_ids(
        local_path, renames=COLUMN_RENAMES, inst_col_pattern=INST_COL_PATTERN
    )
    if not inst_col or not inst_ids:
        logger.warning("No institution IDs for file=%s fp=%s; skipping.", file_name, fp)
        continue
    pending[fp] = (
        {
            "file_name": file_name,
            "local_path": local_path,
            "file_size": row["file_size"],
            "file_modified_time": row["file_modified_time"],
        },
        inst_ids,
        inst_col,
    )

if missing:
    raise FileNotFoundError("Missing staged files: " + "; ".join(missing))
if not pending:
    runtime.notebook_exit(dbutils, f"NO_WORK_ITEMS;BACKFILLED={backfilled}")

sst_by_pdp = resolve_sst_institutions(
    api_client, [pid for _, ids, _ in pending.values() for pid in ids]
)
work_items = [
    {
        "file_fingerprint": fp,
        "file_name": meta["file_name"],
        "local_path": meta["local_path"],
        "institution_id": pdp_id,
        "inst_id": sst_by_pdp[pdp_id][0],
        "institution_name": sst_by_pdp[pdp_id][1],
        "inst_col": inst_col,
        "file_size": meta["file_size"],
        "file_modified_time": meta["file_modified_time"],
        "planned_at": now_ts,
    }
    for fp, (meta, inst_ids, inst_col) in pending.items()
    for pdp_id in inst_ids
]
for fp, (meta, inst_ids, inst_col) in pending.items():
    logger.info(
        "file=%s: %s institution(s) via %s preview=%s",
        meta["file_name"],
        len(inst_ids),
        inst_col,
        [
            {"pdp_id": pid, "inst_id": sst_by_pdp[pid][0], "name": sst_by_pdp[pid][1]}
            for pid in inst_ids[:10]
        ],
    )

n = merge_institution_plan_rows(spark, PLAN_TABLE_PATH, work_items)
logger.info("Wrote/updated %s plan row(s) into %s", n, PLAN_TABLE_PATH)
runtime.notebook_exit(dbutils, f"WORK_ITEMS={n};BACKFILLED={backfilled}")
