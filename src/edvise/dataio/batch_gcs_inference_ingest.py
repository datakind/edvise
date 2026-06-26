"""
Materialize batch-scoped bronze inputs for ES/Legacy inference.

If ``gcs_uploads/{batch_id}/`` already has a completion marker and data files
(upload-time bronze sync), that directory is reused. When not ready yet, poll for
the async bronze sync job before downloading validated objects from GCS into the
same ``gcs_uploads/{batch_id}/`` bronze path.

Cohort/course assignment is deferred to ``es_data_audit`` (edvise_id) or GenAI
execute (``inputs.toml``).
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from google.cloud import storage

from edvise.utils.gcs import (
    DEFAULT_GCS_PREFIX,
    SUCCESS_FILENAME,
    assert_safe_volume_segment,
    copy_validated_blobs_to_landing,
    normalize_gcs_prefix,
    parse_include_blob_paths_json,
    write_success_marker,
)
from edvise.utils.databricks import local_fs_path

LOGGER = logging.getLogger(__name__)

BRONZE_GCS_UPLOADS_SUBDIR = "gcs_uploads"
_DATA_FILE_EXTENSIONS = (".csv", ".parquet")
DEFAULT_BRONZE_SYNC_WAIT_SECONDS = 300.0
DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS = 10.0

IngestSource = Literal["existing_bronze_batch", "gcs_download"]


@dataclass(frozen=True, slots=True)
class BatchIngestResult:
    """Outcome of batch GCS inference ingest."""

    bronze_batch_dir: str
    cohort_dataset_validated_path: Optional[str]
    course_dataset_validated_path: Optional[str]
    source: IngestSource
    copied_count: int
    skipped: bool


def bronze_volume_root(db_workspace: str, institution: str) -> str:
    ws = assert_safe_volume_segment("DB_workspace", db_workspace.strip())
    inst = assert_safe_volume_segment(
        "databricks_institution_name", institution.strip()
    )
    return f"/Volumes/{ws}/{inst}_bronze/bronze_volume"


def bronze_gcs_batch_dir(
    db_workspace: str,
    institution: str,
    batch_id: str,
    *,
    bronze_subdir: str = BRONZE_GCS_UPLOADS_SUBDIR,
) -> str:
    """``bronze_volume/gcs_uploads/{batch_id}/`` landing dir."""
    batch = assert_safe_volume_segment("batch_id", batch_id.strip())
    sub = assert_safe_volume_segment("bronze_subdir", bronze_subdir.strip())
    return os.path.join(bronze_volume_root(db_workspace, institution), sub, batch)


def _is_data_file(path: pathlib.Path) -> bool:
    return path.is_file() and path.suffix.lower() in _DATA_FILE_EXTENSIONS


def is_bronze_batch_ready(batch_dir: str, *, min_files: int = 1) -> bool:
    """
    True when ``batch_dir`` has ``_SUCCESS.json`` and at least ``min_files`` data files.
    """
    base = pathlib.Path(local_fs_path(batch_dir))
    if not base.is_dir():
        return False
    marker = base / SUCCESS_FILENAME
    if not marker.is_file():
        return False
    data_files = [p for p in base.iterdir() if _is_data_file(p)]
    return len(data_files) >= min_files


def wait_for_bronze_batch_ready(
    batch_dir: str,
    *,
    timeout_seconds: float = DEFAULT_BRONZE_SYNC_WAIT_SECONDS,
    poll_interval_seconds: float = DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS,
    min_files: int = 1,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """
    Poll until upload-time bronze sync marks ``batch_dir`` ready or timeout expires.
    """
    if is_bronze_batch_ready(batch_dir, min_files=min_files):
        return True
    if timeout_seconds <= 0:
        return False

    LOGGER.info(
        "Bronze batch dir not ready; waiting up to %s s for upload-time sync: %s",
        timeout_seconds,
        batch_dir,
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_fn(min(poll_interval_seconds, remaining))
        if is_bronze_batch_ready(batch_dir, min_files=min_files):
            LOGGER.info("Bronze batch dir became ready after wait: %s", batch_dir)
            return True

    LOGGER.info(
        "Timed out after %s s waiting for bronze batch dir: %s",
        timeout_seconds,
        batch_dir,
    )
    return False


def parse_is_genai_institution(raw: object) -> bool:
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no", ""):
        return False
    raise ValueError(f"Invalid is_genai_institution value: {raw!r}")


def should_skip_batch_ingest(
    *,
    is_genai_institution: object,
    validated_blob_paths_json: str,
) -> bool:
    """Skip ingest when no validated blob paths were supplied (GenAI and Edvise schema alike)."""
    _ = is_genai_institution  # retained for call-site compatibility
    return len(parse_include_blob_paths_json(validated_blob_paths_json)) == 0


def run_batch_gcs_inference_ingest(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    gcp_bucket_name: str,
    batch_id: str,
    validated_blob_paths_json: str,
    db_run_id: str,
    gcs_source_prefix: str = DEFAULT_GCS_PREFIX,
    require_at_least_one_file: bool = True,
    max_objects: int = 1_000,
    storage_client: Optional[storage.Client] = None,
    is_genai_institution: object = False,
    bronze_sync_wait_seconds: float = DEFAULT_BRONZE_SYNC_WAIT_SECONDS,
    bronze_sync_poll_interval_seconds: float = DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS,
) -> BatchIngestResult:
    """
    Materialize ``bronze_batch_dir`` for downstream tasks.

    Reuses an existing batch bronze folder when ready; waits for upload-time
    bronze sync when not ready; otherwise downloads validated GCS objects into
    ``gcs_uploads/{batch_id}/`` on bronze volume. Cohort/course file assignment
    is deferred to ``es_data_audit`` (edvise_id) or GenAI execute (genai_id).
    """
    if should_skip_batch_ingest(
        is_genai_institution=is_genai_institution,
        validated_blob_paths_json=validated_blob_paths_json,
    ):
        LOGGER.info(
            "Skipping batch GCS inference ingest (blob_paths empty=%s)",
            not parse_include_blob_paths_json(validated_blob_paths_json),
        )
        return BatchIngestResult(
            bronze_batch_dir="",
            cohort_dataset_validated_path=None,
            course_dataset_validated_path=None,
            source="existing_bronze_batch",
            copied_count=0,
            skipped=True,
        )

    if not gcp_bucket_name.strip():
        raise ValueError("gcp_bucket_name is required for batch GCS inference ingest.")

    batch_id_clean = assert_safe_volume_segment("batch_id", batch_id.strip())
    include_paths = parse_include_blob_paths_json(validated_blob_paths_json)
    gcs_prefix = normalize_gcs_prefix(gcs_source_prefix) or DEFAULT_GCS_PREFIX

    existing_dir = bronze_gcs_batch_dir(
        db_workspace, databricks_institution_name, batch_id_clean
    )
    if is_bronze_batch_ready(existing_dir) or wait_for_bronze_batch_ready(
        existing_dir,
        timeout_seconds=bronze_sync_wait_seconds,
        poll_interval_seconds=bronze_sync_poll_interval_seconds,
    ):
        LOGGER.info("Reusing ready bronze batch dir: %s", existing_dir)
        bronze_batch_dir = existing_dir
        source: IngestSource = "existing_bronze_batch"
        copied = 0
    else:
        bronze_batch_dir = existing_dir
        os.makedirs(local_fs_path(bronze_batch_dir), exist_ok=True)
        copied = copy_validated_blobs_to_landing(
            bucket_name=gcp_bucket_name.strip(),
            gcs_prefix=gcs_prefix,
            landing_dir=bronze_batch_dir,
            include_paths=include_paths,
            max_objects=max_objects,
            storage_client=storage_client,
        )
        if copied == 0 and require_at_least_one_file:
            raise FileNotFoundError(
                f"No objects copied from validated blob list under gs://{gcp_bucket_name}/"
            )
        write_success_marker(
            bronze_batch_dir,
            copied=copied,
            bucket_name=gcp_bucket_name.strip(),
            gcs_prefix=gcs_prefix,
            storage_layout="gcs_uploads",
            copy_mode="selective",
            include_blob_paths=include_paths,
        )
        source = "gcs_download"
        LOGGER.info(
            "Downloaded %s validated object(s) to bronze batch dir: %s",
            copied,
            bronze_batch_dir,
        )

    return BatchIngestResult(
        bronze_batch_dir=bronze_batch_dir,
        cohort_dataset_validated_path=None,
        course_dataset_validated_path=None,
        source=source,
        copied_count=copied,
        skipped=False,
    )


def _get_dbutils():
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore[import-not-found]

        return dbutils
    except (ImportError, ModuleNotFoundError):
        return None


def set_batch_ingest_task_values(result: BatchIngestResult) -> None:
    """Publish ingest outputs to Databricks multi-task job values when available."""
    dbc = _get_dbutils()
    if not dbc:
        return

    try:
        dbc.jobs.taskValues.set(
            key="bronze_batch_dir", value=result.bronze_batch_dir or ""
        )
        dbc.jobs.taskValues.set(
            key="cohort_dataset_validated_path",
            value=result.cohort_dataset_validated_path or "",
        )
        dbc.jobs.taskValues.set(
            key="course_dataset_validated_path",
            value=result.course_dataset_validated_path or "",
        )
        dbc.jobs.taskValues.set(
            key="ingest_copied_count", value=str(result.copied_count)
        )
        dbc.jobs.taskValues.set(
            key="ingest_source",
            value=result.source if not result.skipped else "skipped",
        )
    except (AttributeError, OSError, RuntimeError, TypeError) as e:
        LOGGER.error(
            "dbutils.jobs.taskValues.set failed; ingest finished but task values "
            "unavailable: %s",
            e,
            exc_info=True,
        )
