"""
SFTP scan → select cohort/course pair → stage NEW files into pending_ingest_queue.
"""

from __future__ import annotations

from edvise.ingestion.nsc_sftp import runtime

runtime.bootstrap_catalog()

from pyspark.sql import functions as F

from edvise.ingestion.nsc_sftp.constants import (
    MANIFEST_TABLE_PATH,
    QUEUE_TABLE_PATH,
    SFTP_REMOTE_FOLDER,
    SFTP_SOURCE_SYSTEM,
    SFTP_TMP_DIR,
)
from edvise.ingestion.nsc_sftp.file_selection import select_file_pair
from edvise.ingestion.nsc_sftp.helpers import (
    build_listing_df,
    download_new_files_and_queue,
    ensure_manifest_and_queue_tables,
    get_files_to_queue,
    upsert_new_to_manifest,
)
from edvise.utils.sftp import connect_sftp, list_receive_files

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)

asset_scope = "nsc-sftp-asset"
host = dbutils.secrets.get(scope=asset_scope, key="nsc-sftp-host")
user = dbutils.secrets.get(scope=asset_scope, key="nsc-sftp-user")
password = dbutils.secrets.get(scope=asset_scope, key="nsc-sftp-password")

cohort_file_name = runtime.job_param("cohort_file_name")
course_file_name = runtime.job_param("course_file_name")
file_selection_mode = (
    runtime.job_param("file_selection_mode", "uningested").lower() or "uningested"
)

logger.info(
    "Selection inputs: mode=%s cohort=%r course=%r staging=%s",
    file_selection_mode,
    cohort_file_name,
    course_file_name,
    SFTP_TMP_DIR,
)

transport = sftp = None
try:
    ensure_manifest_and_queue_tables(spark)
    transport, sftp = connect_sftp(host, user, password)

    file_rows_all = list_receive_files(sftp, SFTP_REMOTE_FOLDER, SFTP_SOURCE_SYSTEM)
    if not file_rows_all:
        logger.info("No files in %s; exiting.", SFTP_REMOTE_FOLDER)
        dbutils.notebook.exit("NO_FILES")

    available = sorted({r["file_name"] for r in file_rows_all if r.get("file_name")})
    logger.info("SFTP files=%s preview=%s", len(available), available[:25])

    # Cheap uningested check: BRONZE_WRITTEN file names only (no full Spark fingerprint pass).
    ingested_names: set[str] = set()
    if spark.catalog.tableExists(MANIFEST_TABLE_PATH):
        ingested_names = {
            r["file_name"]
            for r in spark.table(MANIFEST_TABLE_PATH)
            .where(F.col("status") == F.lit("BRONZE_WRITTEN"))
            .select("file_name")
            .collect()
            if r["file_name"]
        }

    cohort_file_name, course_file_name, mode_used = select_file_pair(
        file_rows_all,
        mode=file_selection_mode,
        cohort_file_name=cohort_file_name,
        course_file_name=course_file_name,
        ingested_file_names=ingested_names,
    )
    logger.info(
        "Selected via %s: cohort=%s course=%s",
        mode_used,
        cohort_file_name,
        course_file_name,
    )

    requested = {cohort_file_name, course_file_name}
    file_rows = [r for r in file_rows_all if r.get("file_name") in requested]
    missing = sorted(requested - {r.get("file_name") for r in file_rows})
    if missing:
        raise FileNotFoundError(
            f"Requested file(s) missing from {SFTP_REMOTE_FOLDER}: {missing}. "
            f"Available preview={available[:25]}"
        )

    df_listing = build_listing_df(spark, file_rows)
    fingerprints = [
        r.file_fingerprint for r in df_listing.select("file_fingerprint").collect()
    ]
    upsert_new_to_manifest(spark, df_listing)

    df_to_queue = get_files_to_queue(spark, df_listing)
    if df_to_queue.limit(1).count() == 0:
        logger.info("Nothing NEW to queue; exiting.")
        dbutils.notebook.exit("QUEUED_FILES=0")

    queued_count = download_new_files_and_queue(spark, sftp, df_to_queue, logger)
    logger.info(
        "Queued %s file(s). fingerprints=%s table=%s",
        queued_count,
        fingerprints,
        QUEUE_TABLE_PATH,
    )
    dbutils.notebook.exit(f"QUEUED_FILES={queued_count}")
finally:
    for closer in (sftp, transport):
        try:
            if closer is not None:
                closer.close()
        except Exception:
            pass
