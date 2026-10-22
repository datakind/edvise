"""PDP inference inputs under ``gcs_uploads/{batch_id}/``.

When a batch id is present, reuse that folder if both files and ``_SUCCESS.json``
are already there. Otherwise copy ``validated/`` once and write the marker.
Callers without a batch id keep the per-run ``inference_inputs/{run_id}`` path.
"""

from __future__ import annotations

import logging
import os
import pathlib

from edvise.dataio.batch_gcs_inference_ingest import bronze_gcs_batch_dir
from edvise.utils.databricks import local_fs_path
from edvise.utils.gcs import (
    DEFAULT_GCS_PREFIX,
    SUCCESS_FILENAME,
    copy_validated_blobs_to_landing,
    write_success_marker,
)

LOGGER = logging.getLogger(__name__)


def pdp_batch_inputs_ready(
    batch_dir: str, cohort_file_name: str, course_file_name: str
) -> bool:
    """True when the batch folder has the marker and both named input files."""
    base = pathlib.Path(local_fs_path(batch_dir))
    if not (base / SUCCESS_FILENAME).is_file():
        return False
    return (base / cohort_file_name).is_file() and (base / course_file_name).is_file()


def materialize_pdp_batch_inputs(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    gcp_bucket_name: str,
    batch_id: str,
    cohort_file_name: str,
    course_file_name: str,
    storage_client: object | None = None,
) -> tuple[str, str, str]:
    """Return ``(course_path, cohort_path, batch_dir)`` for one PDP batch.

    Downloads from GCS only when the batch folder is not already complete.
    """
    batch_dir = bronze_gcs_batch_dir(
        db_workspace, databricks_institution_name, batch_id
    )
    course_path = os.path.join(batch_dir, course_file_name)
    cohort_path = os.path.join(batch_dir, cohort_file_name)
    if pdp_batch_inputs_ready(batch_dir, cohort_file_name, course_file_name):
        LOGGER.info("Reusing PDP batch inputs at %s", batch_dir)
        return course_path, cohort_path, batch_dir

    include_paths = [
        f"{DEFAULT_GCS_PREFIX}{cohort_file_name}",
        f"{DEFAULT_GCS_PREFIX}{course_file_name}",
    ]
    os.makedirs(local_fs_path(batch_dir), exist_ok=True)
    copied = copy_validated_blobs_to_landing(
        bucket_name=gcp_bucket_name,
        gcs_prefix=DEFAULT_GCS_PREFIX,
        landing_dir=batch_dir,
        include_paths=include_paths,
        max_objects=len(include_paths),
        storage_client=storage_client,  # type: ignore[arg-type]
    )
    if copied != len(include_paths):
        raise FileNotFoundError(
            f"Expected {len(include_paths)} validated objects in "
            f"gs://{gcp_bucket_name}/{DEFAULT_GCS_PREFIX}, copied {copied}."
        )
    write_success_marker(
        batch_dir,
        copied=copied,
        bucket_name=gcp_bucket_name,
        gcs_prefix=DEFAULT_GCS_PREFIX,
        storage_layout="gcs_uploads",
        copy_mode="selective",
        include_blob_paths=include_paths,
    )
    LOGGER.info("Backfilled %s validated object(s) to %s", copied, batch_dir)
    return course_path, cohort_path, batch_dir
