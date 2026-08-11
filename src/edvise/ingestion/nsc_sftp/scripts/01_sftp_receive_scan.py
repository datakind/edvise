"""
SFTP scan → select cohort/course pair → stage NEW files into pending_ingest_queue.
"""

from __future__ import annotations

import os
import sys

# Databricks GIT spark_python_task: put repo src/ on path before importing edvise.
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

from edvise.ingestion.nsc_sftp.constants import (
    QUEUE_TABLE_PATH,
    SFTP_REMOTE_FOLDER,
    SFTP_SECRET_KEY_HOST,
    SFTP_SECRET_KEY_PASSWORD,
    SFTP_SECRET_KEY_USER,
    SFTP_SOURCE_SYSTEM,
    SFTP_TMP_DIR,
)
from edvise.ingestion.nsc_sftp.file_selection import select_file_pair
from edvise.ingestion.nsc_sftp.helpers import (
    bronze_written_file_names,
    build_listing_df,
    download_new_files_and_queue,
    ensure_manifest_and_queue_tables,
    get_files_to_queue,
    reset_files_for_reingest,
    upsert_new_to_manifest,
)
from edvise.utils.sftp import connect_sftp, list_receive_files

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)

secret_scope = runtime.require_job_param("nsc_sftp_secret_scope")
host = dbutils.secrets.get(scope=secret_scope, key=SFTP_SECRET_KEY_HOST)
user = dbutils.secrets.get(scope=secret_scope, key=SFTP_SECRET_KEY_USER)
password = dbutils.secrets.get(scope=secret_scope, key=SFTP_SECRET_KEY_PASSWORD)

cohort_file_name = runtime.job_param("cohort_file_name")
course_file_name = runtime.job_param("course_file_name")
file_selection_mode = (
    runtime.job_param("file_selection_mode", "skip_ingested").lower() or "skip_ingested"
)
force_reingest = runtime.job_param_bool("force_reingest", False)

logger.info(
    "Selection inputs: mode=%s force_reingest=%s cohort=%r course=%r staging=%s",
    file_selection_mode,
    force_reingest,
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
        runtime.notebook_exit(dbutils, "NO_FILES")

    available = sorted({r["file_name"] for r in file_rows_all if r.get("file_name")})
    logger.info("SFTP files=%s preview=%s", len(available), available[:25])

    cohort_file_name, course_file_name, mode_used = select_file_pair(
        file_rows_all,
        mode=file_selection_mode,
        cohort_file_name=cohort_file_name,
        course_file_name=course_file_name,
        # force_reingest: allow selecting pairs already BRONZE_WRITTEN
        ingested_file_names=set()
        if force_reingest
        else bronze_written_file_names(spark),
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
    if force_reingest:
        reset_files_for_reingest(
            spark,
            [
                str(r["file_fingerprint"])
                for r in df_listing.select("file_fingerprint").collect()
            ],
        )
    upsert_new_to_manifest(spark, df_listing)

    df_to_queue = get_files_to_queue(spark, df_listing)
    if df_to_queue.limit(1).count() == 0:
        logger.info("Nothing NEW to queue; exiting.")
        runtime.notebook_exit(dbutils, "QUEUED_FILES=0")

    queued_count = download_new_files_and_queue(spark, sftp, df_to_queue, logger)
    logger.info("Queued %s file(s) into %s", queued_count, QUEUE_TABLE_PATH)
    runtime.notebook_exit(
        dbutils, f"QUEUED_FILES={queued_count};FORCE_REINGEST={force_reingest}"
    )
finally:
    for closer in (sftp, transport):
        try:
            if closer is not None:
                closer.close()
        except Exception:
            pass
