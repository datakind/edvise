"""
Connect to SFTP, scan the receive folder, upsert unseen files into ingestion_manifest,
and stage NEW files into pending_ingest_queue.

Requires SFTP secrets and job parameters cohort_file_name / course_file_name.
Cluster image must include paramiko (and other deps) — see pyproject optional/install docs.

Outputs:
  - Delta: ingestion_manifest, pending_ingest_queue
  - Staged files under UC volume path from nsc_sftp.constants.SFTP_TMP_DIR
"""

from __future__ import annotations

import logging
import os
import re
import sys

from databricks.connect import DatabricksSession
from pyspark.sql import functions as F

from edvise import utils
from edvise.ingestion.nsc_sftp.constants import (
    MANIFEST_TABLE_PATH,
    QUEUE_TABLE_PATH,
    SFTP_REMOTE_FOLDER,
    SFTP_SOURCE_SYSTEM,
    SFTP_TMP_DIR,
)
from edvise.ingestion.nsc_sftp.helpers import (
    build_listing_df,
    download_new_files_and_queue,
    ensure_manifest_and_queue_tables,
    get_files_to_queue,
    upsert_new_to_manifest,
)
from edvise.utils.sftp import connect_sftp, list_receive_files


def _spark_python_task_args(argv: list[str]) -> dict[str, str]:
    """Parse ``--key value`` pairs from ``spark_python_task.parameters``."""
    out: dict[str, str] = {}
    i = 0
    args = argv[1:]
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args):
            out[a[2:].replace("-", "_")] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


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

asset_scope = "nsc-sftp-asset"

host = dbutils.secrets.get(scope=asset_scope, key="nsc-sftp-host")
user = dbutils.secrets.get(scope=asset_scope, key="nsc-sftp-user")
password = dbutils.secrets.get(scope=asset_scope, key="nsc-sftp-password")

_argv = _spark_python_task_args(sys.argv)
cohort_file_name = utils.databricks.get_db_widget_param(
    "cohort_file_name", default=_argv.get("cohort_file_name", "")
)
course_file_name = utils.databricks.get_db_widget_param(
    "course_file_name", default=_argv.get("course_file_name", "")
)
cohort_file_name = str(cohort_file_name).strip()
course_file_name = str(course_file_name).strip()
if not cohort_file_name or not course_file_name:
    raise ValueError(
        "Missing required workflow parameters: cohort_file_name and course_file_name. "
        "Pass them as Databricks job base parameters."
    )


def _extract_file_stamp(file_name: str) -> str:
    base = os.path.basename(file_name)
    m = re.search(r"_(\d{14})(?:\.[^.]+)?$", base)
    if not m:
        raise ValueError(
            "Expected file name to end with a 14-digit file stamp, e.g. "
            "'..._YYYYMMDDHHMMSS.csv'. Got: "
            f"{file_name}"
        )
    return m.group(1)


cohort_stamp = _extract_file_stamp(cohort_file_name)
course_stamp = _extract_file_stamp(course_file_name)
if cohort_stamp != course_stamp:
    raise ValueError(
        "cohort_file_name and course_file_name must end with the same file stamp. "
        f"Got cohort stamp={cohort_stamp}, course stamp={course_stamp}."
    )
logger.info(f"Validated file stamp: {cohort_stamp}")
logger.info(f"Staging to UC volume path: {SFTP_TMP_DIR}")
logger.info(
    "Manual file selection enabled: "
    f"cohort_file_name={cohort_file_name}, course_file_name={course_file_name}"
)

logger.info("SFTP secured assets loaded successfully.")

transport = None
sftp = None

try:
    ensure_manifest_and_queue_tables(spark)

    transport, sftp = connect_sftp(host, user, password)
    logger.info(
        f"Connected to SFTP host={host} and scanning folder={SFTP_REMOTE_FOLDER}"
    )

    file_rows_all = list_receive_files(sftp, SFTP_REMOTE_FOLDER, SFTP_SOURCE_SYSTEM)
    if not file_rows_all:
        logger.info(
            f"No files found in SFTP folder: {SFTP_REMOTE_FOLDER}. Exiting (no-op)."
        )
        dbutils.notebook.exit("NO_FILES")

    requested_names = {cohort_file_name, course_file_name}
    logger.info(
        f"Found {len(file_rows_all)} file(s) on SFTP in folder={SFTP_REMOTE_FOLDER}; "
        f"requested={sorted(requested_names)}"
    )
    file_rows = [r for r in file_rows_all if r.get("file_name") in requested_names]

    found_names = {r.get("file_name") for r in file_rows}
    missing_names = sorted(requested_names - found_names)
    if missing_names:
        available = sorted({r.get("file_name") for r in file_rows_all})
        preview = available[:25]
        raise FileNotFoundError(
            f"Requested file(s) not found on SFTP in folder '{SFTP_REMOTE_FOLDER}': {missing_names}. "
            f"Available file count={len(available)}; first 25={preview}"
        )

    for r in file_rows:
        logger.info(
            f"Selected SFTP file: name={r.get('file_name')} size={r.get('file_size')} "
            f"modified={r.get('file_modified_time')}"
        )

    df_listing = build_listing_df(spark, file_rows)
    fingerprints = [
        r["file_fingerprint"] for r in df_listing.select("file_fingerprint").collect()
    ]

    logger.info("SFTP listing (selected files):")
    df_listing.select(
        "file_name", "file_size", "file_modified_time", "file_fingerprint"
    ).show(truncate=False)

    upsert_new_to_manifest(spark, df_listing)

    logger.info("Manifest rows (selected files):")
    spark.table(MANIFEST_TABLE_PATH).where(
        F.col("file_fingerprint").isin(fingerprints)
    ).select(
        "file_name",
        "file_fingerprint",
        "status",
        "processed_at",
        "error_message",
    ).show(truncate=False)

    df_to_queue = get_files_to_queue(spark, df_listing)

    to_queue_count = df_to_queue.count()
    if to_queue_count == 0:
        logger.info(
            "No files to queue: either nothing is NEW, or NEW files are already queued. Exiting (no-op)."
        )
        dbutils.notebook.exit("QUEUED_FILES=0")

    logger.info("Files eligible to queue:")
    df_to_queue.select(
        "file_name", "file_size", "file_modified_time", "file_fingerprint"
    ).show(truncate=False)

    logger.info(
        f"Queuing {to_queue_count} NEW-unqueued file(s) to {QUEUE_TABLE_PATH} and staging to UC volume."
    )
    queued_count = download_new_files_and_queue(spark, sftp, df_to_queue, logger)

    logger.info("Queue rows (selected files):")
    spark.table(QUEUE_TABLE_PATH).where(
        F.col("file_fingerprint").isin(fingerprints)
    ).select("file_name", "file_fingerprint", "local_tmp_path", "queued_at").show(
        truncate=False
    )

    logger.info(
        f"Queued {queued_count} file(s) for downstream processing in {QUEUE_TABLE_PATH}."
    )
    dbutils.notebook.exit(f"QUEUED_FILES={queued_count}")

finally:
    try:
        if sftp is not None:
            sftp.close()
    except Exception:
        pass
    try:
        if transport is not None:
            transport.close()
    except Exception:
        pass
