"""
Register SPLIT_STAGED cohort/course pairs as webapp batches.

Copies validated GCS objects into bronze_volume/gcs_uploads/{batch_id} and
marks the SFTP manifest BRONZE_WRITTEN only after every on-disk split for that
file is in a materialized batch. Staging files in default/tmp are left in place.
"""

from __future__ import annotations

import os
import re
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
    COLUMN_RENAMES,
    DEFAULT_SCHEMA,
    INSTITUTION_COLUMN_PATTERN,
    MANIFEST_TABLE_PATH,
    PLAN_TABLE_PATH,
    SFTP_TMP_DIR,
)
from edvise.ingestion.nsc_sftp.file_selection import extract_file_stamp  # noqa: E402
from edvise.ingestion.nsc_sftp.helpers import (  # noqa: E402
    extract_institution_ids,
    group_dataframe_by_institution_id,
    load_staged_csv,
    log_section,
    process_and_save_file,
    resolve_bronze_volume_dir,
    resolve_sst_institution,
    update_manifest,
)
from edvise.ingestion.nsc_sftp.sftp import output_file_name_from_sftp  # noqa: E402
from edvise.ingestion.nsc_sftp.webapp_batch import (  # noqa: E402
    InstitutionPair,
    build_registration_only_pair,
    fingerprints_ready_to_close,
    pair_plan_rows,
    register_pair,
    select_tmp_registration_pair,
    shared_pdp_ids,
    split_csv_path,
    tmp_original_file_name,
)
from edvise.utils.gcs import assert_safe_volume_segment  # noqa: E402
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


def _register_one(pair: InstitutionPair) -> dict[str, str]:
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
        bronze_dir = resolve_bronze_volume_dir(spark, CATALOG, prefix)
        result = register_pair(
            api_client,
            pair,
            db_workspace=db_workspace,
            bronze_dir=bronze_dir,
        )
    except Exception as exc:
        base_row["error_message"] = str(exc)[:8000]
        _upsert_registration(base_row)
        raise
    base_row.update(result)
    base_row["error_message"] = ""
    _upsert_registration(base_row)
    return result


def _one_school_slice(source_path: str, pdp_id: str, dest_dir: str) -> str:
    """Write one PDP id's rows from a tmp download. Upload name drops the fingerprint."""
    inst_col_pattern = re.compile(INSTITUTION_COLUMN_PATTERN, re.IGNORECASE)
    inst_col, _ids = extract_institution_ids(
        source_path, renames=COLUMN_RENAMES, inst_col_pattern=inst_col_pattern
    )
    if not inst_col:
        raise ValueError(f"No institution id column in {source_path}")
    frame = load_staged_csv(source_path, renames=COLUMN_RENAMES, inst_col=inst_col)
    grouped = group_dataframe_by_institution_id(frame, inst_col, [pdp_id])
    filtered = grouped.get(pdp_id)
    if filtered is None or filtered.empty:
        raise ValueError(f"No rows for pdp_id={pdp_id} in {source_path}")
    out_name = output_file_name_from_sftp(tmp_original_file_name(source_path))
    return process_and_save_file(dest_dir, out_name, filtered)


if runtime.job_param_bool("registration_only", False):
    downloads = [
        os.path.join(SFTP_TMP_DIR, name)
        for name in os.listdir(SFTP_TMP_DIR)
        if os.path.isfile(os.path.join(SFTP_TMP_DIR, name))
    ]
    cohort_csv_path, course_csv_path = select_tmp_registration_pair(downloads)
    inst_col_pattern = re.compile(INSTITUTION_COLUMN_PATTERN, re.IGNORECASE)
    _, cohort_ids = extract_institution_ids(
        cohort_csv_path, renames=COLUMN_RENAMES, inst_col_pattern=inst_col_pattern
    )
    _, course_ids = extract_institution_ids(
        course_csv_path, renames=COLUMN_RENAMES, inst_col_pattern=inst_col_pattern
    )
    pdp_ids = shared_pdp_ids(cohort_ids, course_ids)
    logger.info(
        "registration_only files cohort=%s course=%s; registering %s pdp_id(s): %s",
        cohort_csv_path,
        course_csv_path,
        len(pdp_ids),
        pdp_ids,
    )
    stamp = extract_file_stamp(tmp_original_file_name(cohort_csv_path))
    slice_dir = os.path.join(SFTP_TMP_DIR, "registration_only", stamp)
    os.makedirs(slice_dir, exist_ok=True)
    _ensure_registration_table()
    written: list[str] = []
    failed: list[str] = []
    for pdp_id in pdp_ids:
        school_dir = os.path.join(
            slice_dir, assert_safe_volume_segment("pdp_id", pdp_id)
        )
        os.makedirs(school_dir, exist_ok=True)
        try:
            inst_id, institution_name = resolve_sst_institution(api_client, pdp_id)
            pair = build_registration_only_pair(
                cohort_csv_path=_one_school_slice(cohort_csv_path, pdp_id, school_dir),
                course_csv_path=_one_school_slice(course_csv_path, pdp_id, school_dir),
                inst_id=inst_id,
                pdp_id=pdp_id,
                institution_name=institution_name,
            )
            logger.info(
                "registration_only: inst_id=%s pdp_id=%s stamp=%s cohort=%s course=%s",
                pair.inst_id,
                pair.pdp_id,
                pair.stamp,
                pair.cohort_split_path,
                pair.course_split_path,
            )
            result = _register_one(pair)
        except Exception as exc:
            failed.append(f"pdp_id={pdp_id}: {exc}")
            logger.exception("registration_only failed for pdp_id=%s", pdp_id)
            continue
        written.append(
            f"pdp_id={pair.pdp_id} {pair.institution_name} "
            f"batch_id={result['batch_id']} batch_name={result['batch_name']} "
            f"-> {result['bronze_path']}"
        )
        logger.info(
            "Registered pdp_id=%s %s batch_id=%s batch_name=%s -> %s",
            pair.pdp_id,
            pair.institution_name,
            result["batch_id"],
            result["batch_name"],
            result["bronze_path"],
        )
    log_section(logger, "Registered", written)
    log_section(logger, "Failed", failed)
    if failed:
        raise RuntimeError(
            f"registration_only registered {len(written)} of {len(pdp_ids)}. "
            + " | ".join(failed)
        )
    runtime.notebook_exit(
        dbutils,
        f"REGISTERED={len(written)};" + ";".join(written),
    )

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
    runtime.notebook_exit(dbutils, "NO_SPLIT_STAGED")

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
runtime.notebook_exit(
    dbutils,
    f"REGISTERED={len(written)};FAILED={len(failed)};UNPAIRED={len(problems)}",
)
