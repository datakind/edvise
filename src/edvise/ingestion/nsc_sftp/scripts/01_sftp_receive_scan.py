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
    log_section,
    reset_files_for_reingest,
    upsert_new_to_manifest,
)
from edvise.utils.sftp import connect_sftp, list_receive_files

dbutils = runtime.get_dbutils()
spark = runtime.get_spark()
logger = runtime.get_logger(__name__)


def _stage01_exit_message(
    *,
    queued_count: int,
    force_reingest: bool,
    mode_used: str,
    cohort_file_name: str | None,
    course_file_name: str | None,
    queued_names: str,
    available: list[str],
) -> str:
    """Multiline summary for the Databricks task Output panel."""
    sftp_block = "\n".join(f"  {name}" for name in available) or "  (none)"
    files_block = (
        "\n".join(
            f"  {name}" for name in queued_names.split(",") if name and name != "None"
        )
        or "  (none)"
    )
    return (
        f"QUEUED_FILES={queued_count}\n"
        f"FORCE_REINGEST={force_reingest}\n"
        f"MODE={mode_used}\n"
        f"COHORT={cohort_file_name or 'None'}\n"
        f"COURSE={course_file_name or 'None'}\n"
        f"FILES ({queued_count}):\n{files_block}\n"
        f"SFTP_FILE_COUNT={len(available)}\n"
        f"SFTP_FILES:\n{sftp_block}"
    )


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
    "Stage 01 — SFTP scan: mode=%s force_reingest=%s staging=%s",
    file_selection_mode,
    force_reingest,
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
    log_section(logger, "SFTP files", available)

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
            f"Available={available}"
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
        logger.info("Nothing new to queue; exiting.")
        runtime.notebook_exit(
            dbutils,
            _stage01_exit_message(
                queued_count=0,
                force_reingest=force_reingest,
                mode_used=mode_used,
                cohort_file_name=cohort_file_name,
                course_file_name=course_file_name,
                queued_names="None",
                available=available,
            ),
        )

    # Collect metadata before download: after queue upsert, left_anti makes
    # df_to_queue empty if we collect again.
    queued_rows = df_to_queue.select(
        "file_name", "file_fingerprint", "sftp_path", "file_size"
    ).collect()
    queued_count = download_new_files_and_queue(spark, sftp, df_to_queue, logger)
    log_section(
        logger,
        "Queued for expansion",
        [f"{r['file_name']} (size={r['file_size']})" for r in queued_rows],
    )
    logger.info(
        "Stage 01 done — queued %s file(s) into %s", queued_count, QUEUE_TABLE_PATH
    )
    queued_names = ",".join(str(r["file_name"]) for r in queued_rows) or "None"
    # notebook_exit is what Databricks shows in the task Output panel
    runtime.notebook_exit(
        dbutils,
        _stage01_exit_message(
            queued_count=queued_count,
            force_reingest=force_reingest,
            mode_used=mode_used,
            cohort_file_name=cohort_file_name,
            course_file_name=course_file_name,
            queued_names=queued_names,
            available=available,
        ),
    )
finally:
    for closer in (sftp, transport):
        try:
            if closer is not None:
                closer.close()
        except Exception:
            pass
