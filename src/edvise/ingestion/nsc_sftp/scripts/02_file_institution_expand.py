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
    log_section,
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

logger.info("Stage 02 — expand queued files into institution plan")

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
    logger.info("No new files to expand (all already in plan).")
    runtime.notebook_exit(dbutils, f"NO_NEW_EXPANSION_WORK;BACKFILLED={backfilled}")

queued_files = queue_df.select(
    "file_fingerprint",
    "file_name",
    F.col("local_tmp_path").alias("local_path"),
    "file_size",
    "file_modified_time",
).collect()

logger.info("Expanding %s queued file(s)", len(queued_files))

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
        logger.warning("No institution IDs in file=%s; skipping.", file_name)
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
    logger.info(
        "File %s: found %s unique PDP id(s) in column %s: %s",
        file_name,
        len(inst_ids),
        inst_col,
        ",".join(inst_ids[:20]) + ("..." if len(inst_ids) > 20 else ""),
    )

if missing:
    raise FileNotFoundError("Missing staged files: " + "; ".join(missing))
if not pending:
    runtime.notebook_exit(dbutils, f"NO_WORK_ITEMS;BACKFILLED={backfilled}")

all_pdp_ids = [pid for _, ids, _ in pending.values() for pid in ids]
sst_by_pdp = resolve_sst_institutions(api_client, all_pdp_ids)
unresolved = sorted(
    {str(p).strip() for p in all_pdp_ids if str(p).strip()} - set(sst_by_pdp)
)

planned_unique: dict[str, str] = {}
unresolved_lines: list[str] = []
work_items: list[dict[str, Any]] = []
for fp, (meta, inst_ids, inst_col) in pending.items():
    for pdp_id in inst_ids:
        if pdp_id not in sst_by_pdp:
            unresolved_lines.append(
                f"pdp_id={pdp_id} file={meta['file_name']} (not in SST)"
            )
            continue
        inst_id, inst_name = sst_by_pdp[pdp_id]
        work_items.append(
            {
                "file_fingerprint": fp,
                "file_name": meta["file_name"],
                "local_path": meta["local_path"],
                "institution_id": pdp_id,
                "inst_id": inst_id,
                "institution_name": inst_name,
                "inst_col": inst_col,
                "file_size": meta["file_size"],
                "file_modified_time": meta["file_modified_time"],
                "planned_at": now_ts,
            }
        )
        planned_unique[str(pdp_id)] = str(inst_name)

log_section(
    logger,
    "Institutions planned for bronze",
    [f"pdp_id={pdp_id} {name}" for pdp_id, name in sorted(planned_unique.items())],
)
if unresolved_lines:
    log_section(logger, "Unresolved PDP ids (skipped)", sorted(set(unresolved_lines)))

if not work_items:
    logger.error(
        "No institutions resolved via SST API (unresolved=%s). Nothing to plan.",
        len(unresolved),
    )
    unresolved_summary = ",".join(unresolved[:20]) or "None"
    runtime.notebook_exit(
        dbutils,
        f"WORK_ITEMS=0;UNRESOLVED={len(unresolved)};BACKFILLED={backfilled};"
        f"PLANNED=None;UNRESOLVED_PDP_IDS={unresolved_summary}",
    )

n = merge_institution_plan_rows(spark, PLAN_TABLE_PATH, work_items)
planned_summary = (
    "|".join(f"{pdp_id}:{name}" for pdp_id, name in sorted(planned_unique.items()))
    or "None"
)
unresolved_summary = ",".join(unresolved[:20]) or "None"
logger.info(
    "Stage 02 done — %s plan row(s), %s institution(s), %s unresolved",
    n,
    len(planned_unique),
    len(unresolved),
)
runtime.notebook_exit(
    dbutils,
    f"WORK_ITEMS={n};UNRESOLVED={len(unresolved)};BACKFILLED={backfilled};"
    f"PLANNED={planned_summary};UNRESOLVED_PDP_IDS={unresolved_summary}",
)
