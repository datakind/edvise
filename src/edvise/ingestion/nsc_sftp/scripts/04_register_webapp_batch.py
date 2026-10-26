"""
Register SPLIT_STAGED cohort/course pairs as webapp batches.

Copies validated GCS objects into bronze_volume/gcs_uploads/{batch_id} and
marks the SFTP manifest BRONZE_WRITTEN only after every on-disk split for that
file is in a materialized batch. After that status is written, the fingerprint's
SFTP download and splits directory are removed from the staging volume.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

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

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

from edvise.ingestion.nsc_sftp.constants import (  # noqa: E402
    CATALOG,
    DEFAULT_SCHEMA,
    MANIFEST_TABLE_PATH,
    PLAN_TABLE_PATH,
    SFTP_TMP_DIR,
)
from edvise.ingestion.nsc_sftp.file_selection import extract_file_stamp  # noqa: E402
from edvise.ingestion.nsc_sftp.helpers import (  # noqa: E402
    log_section,
    resolve_bronze_volume_dir,
    update_manifest,
)
from edvise.ingestion.nsc_sftp.webapp_batch import (  # noqa: E402
    fingerprints_ready_to_close,
    pair_plan_rows,
    register_pair,
    remove_staged_fingerprint,
    split_csv_path,
)
from edvise.utils.institution_naming import databricksify_inst_name  # noqa: E402

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)

REGISTRATION_TABLE = f"{CATALOG}.{DEFAULT_SCHEMA}.institution_batch_registration"
REGISTRATION_COLUMNS = [
    "inst_id",
    "stamp",
    "pdp_id",
    "institution_name",
    "cohort_file_name",
    "course_file_name",
    "cohort_fingerprint",
    "course_fingerprint",
    "batch_id",
    "batch_name",
    "bronze_path",
    "status",
    "error_message",
    "updated_at",
]

api_client = runtime.require_edvise_api_client(dbutils)
db_workspace = runtime.require_job_param("DB_workspace")


def _ensure_registration_table() -> None:
    columns = ",\n".join(f"{name} STRING" for name in REGISTRATION_COLUMNS)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {REGISTRATION_TABLE} (
          {columns}
        )
        USING DELTA
        """
    )


def _upsert_registration(row: dict[str, str]) -> None:
    schema = T.StructType(
        [T.StructField(name, T.StringType(), True) for name in REGISTRATION_COLUMNS]
    )
    spark.createDataFrame([row], schema=schema).createOrReplaceTempView(
        "nsc_batch_reg_update"
    )
    assignments = ", ".join(f"t.{name} = s.{name}" for name in REGISTRATION_COLUMNS)
    insert_cols = ", ".join(REGISTRATION_COLUMNS)
    insert_vals = ", ".join(f"s.{name}" for name in REGISTRATION_COLUMNS)
    spark.sql(
        f"""
        MERGE INTO {REGISTRATION_TABLE} t
        USING nsc_batch_reg_update s
        ON t.inst_id = s.inst_id AND t.stamp = s.stamp
        WHEN MATCHED THEN UPDATE SET {assignments}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """
    )


def _materialized_keys() -> set[tuple[str, str, str]]:
    if not spark.catalog.tableExists(REGISTRATION_TABLE):
        return set()
    rows = (
        spark.table(REGISTRATION_TABLE)
        .where(F.col("status") == F.lit("BRONZE_MATERIALIZED"))
        .select(
            "inst_id",
            "stamp",
            "cohort_fingerprint",
            "course_fingerprint",
        )
        .collect()
    )
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        inst_id = str(row["inst_id"] or "")
        stamp = str(row["stamp"] or "")
        for fingerprint in (row["cohort_fingerprint"], row["course_fingerprint"]):
            if inst_id and stamp and fingerprint:
                keys.add((inst_id, stamp, str(fingerprint)))
    return keys


def _remove_bronze_written_staging() -> tuple[list[str], list[str]]:
    """Delete staging files for fingerprints already marked BRONZE_WRITTEN.

    Runs after the manifest update, and again on a later run if a delete failed.
    """
    rows = (
        spark.table(MANIFEST_TABLE_PATH)
        .where(F.col("status") == F.lit("BRONZE_WRITTEN"))
        .select("file_fingerprint", "file_name")
        .distinct()
        .collect()
    )
    removed: list[str] = []
    errors: list[str] = []
    for row in rows:
        fingerprint = str(row["file_fingerprint"] or "")
        file_name = str(row["file_name"] or "")
        if not fingerprint:
            continue
        try:
            removed.extend(
                remove_staged_fingerprint(SFTP_TMP_DIR, fingerprint, file_name)
            )
        except Exception as exc:
            errors.append(f"{fingerprint}: {exc}")
            logger.exception("failed to remove staged files for %s", fingerprint)
    return removed, errors


def _log_staging_cleanup(removed: list[str], errors: list[str]) -> str:
    if removed:
        log_section(logger, "Removed staging", removed)
    if errors:
        log_section(logger, "Staging cleanup failed", errors)
    return f"REMOVED={len(removed)};CLEANUP_FAILED={len(errors)}"


if not spark.catalog.tableExists(PLAN_TABLE_PATH):
    runtime.notebook_exit(dbutils, "NO_PLAN_TABLE")
if not spark.catalog.tableExists(MANIFEST_TABLE_PATH):
    runtime.notebook_exit(dbutils, "NO_MANIFEST_TABLE")

_ensure_registration_table()
staged = (
    spark.table(MANIFEST_TABLE_PATH)
    .where(F.col("status") == F.lit("SPLIT_STAGED"))
    .select("file_fingerprint")
)
plan_rows = (
    spark.table(PLAN_TABLE_PATH)
    .join(staged, on="file_fingerprint", how="inner")
    .select(
        "file_fingerprint",
        "file_name",
        "institution_id",
        "inst_id",
        "institution_name",
    )
    .collect()
)
if not plan_rows:
    cleanup = _log_staging_cleanup(*_remove_bronze_written_staging())
    runtime.notebook_exit(dbutils, f"NO_SPLIT_STAGED;{cleanup}")

row_dicts = [row.asDict() for row in plan_rows]
pairs, problems = pair_plan_rows(row_dicts, tmp_dir=SFTP_TMP_DIR)
log_section(logger, "Unpaired", problems)

bronze_dir_cache: dict[str, str] = {}
written: list[str] = []
failed: list[str] = []
for pair in pairs:
    prefix = databricksify_inst_name(pair.institution_name)
    now = datetime.now(timezone.utc).isoformat()
    base_row = {
        "inst_id": pair.inst_id,
        "stamp": pair.stamp,
        "pdp_id": pair.pdp_id,
        "institution_name": pair.institution_name,
        "cohort_file_name": pair.cohort_file_name,
        "course_file_name": pair.course_file_name,
        "cohort_fingerprint": pair.cohort_fingerprint,
        "course_fingerprint": pair.course_fingerprint,
        "batch_id": "",
        "batch_name": "",
        "bronze_path": "",
        "status": "FAILED",
        "error_message": "",
        "updated_at": now,
    }
    try:
        if prefix not in bronze_dir_cache:
            bronze_dir_cache[prefix] = resolve_bronze_volume_dir(spark, CATALOG, prefix)
        result = register_pair(
            api_client,
            pair,
            db_workspace=db_workspace,
            bronze_dir=bronze_dir_cache[prefix],
        )
        base_row.update(result)
        base_row["error_message"] = ""
        written.append(
            f"pdp_id={pair.pdp_id} {pair.institution_name} "
            f"batch_id={result['batch_id']} -> {result['bronze_path']}"
        )
    except Exception as exc:
        base_row["error_message"] = str(exc)[:8000]
        failed.append(f"pdp_id={pair.pdp_id} stamp={pair.stamp}: {exc}")
        logger.exception("register_pair failed for %s", pair.inst_id)
    _upsert_registration(base_row)

close_rows = []
for row in row_dicts:
    try:
        stamp = extract_file_stamp(str(row.get("file_name") or ""))
    except ValueError:
        continue
    split_path = split_csv_path(
        SFTP_TMP_DIR,
        str(row.get("file_fingerprint") or ""),
        str(row.get("institution_id") or ""),
        str(row.get("file_name") or ""),
    )
    close_rows.append(
        {
            "file_fingerprint": row.get("file_fingerprint"),
            "inst_id": row.get("inst_id"),
            "stamp": stamp,
            "split_exists": os.path.exists(split_path),
        }
    )
for fingerprint in fingerprints_ready_to_close(close_rows, _materialized_keys()):
    update_manifest(
        spark,
        MANIFEST_TABLE_PATH,
        fingerprint,
        status="BRONZE_WRITTEN",
        error_message=None,
        run_id=runtime.workflow_run_id(dbutils),
    )

log_section(logger, "Registered", written)
log_section(logger, "Failed", failed)
cleanup = _log_staging_cleanup(*_remove_bronze_written_staging())
runtime.notebook_exit(
    dbutils,
    f"REGISTERED={len(written)};FAILED={len(failed)};UNPAIRED={len(problems)};{cleanup}",
)
