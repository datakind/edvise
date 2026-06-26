"""
Utility functions that interact with Google Cloud Storage.

Includes single-object download/upload helpers, source resolution for pipeline
inputs, and batch copy of validated objects into local landing directories
(bronze volume paths).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlparse

import google.auth
from google.api_core import exceptions as gax_exc
from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import storage
from google.cloud.storage import Blob

from edvise.utils.databricks import get_dbutils_or_none, local_fs_path

# --- Landing copy constants (bronze sync / batch inference ingest) ---

_SAFE_VOLUME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

SUCCESS_FILENAME = "_SUCCESS.json"
DEFAULT_GCS_PREFIX = "validated/"
MAX_BLOBS_LISTED_IN_SUCCESS_JSON = 50
BLOB_DOWNLOAD_MAX_ATTEMPTS = 3
BLOB_DOWNLOAD_RETRY_DELAY_INITIAL = 0.5
BLOB_DOWNLOAD_RETRY_DELAY_CAP = 4.0
HTTP_STATUS_TRANSIENT_GCS: Tuple[int, ...] = (429, 500, 502, 503, 504)


def save_file(
    bucket: "storage.Bucket", src_volume_filepath: str, dest_bucket_pathname: str
) -> None:
    """Save file from databricks volume to a GCP bucket path.

    Args:
      bucket: The bucket object.
      src_volume_filepath: The source filepath to the Databricks volume.
      dest_bucket_pathname: The destination filepath in GCP.

    Returns:
      Nothing.
    """
    blob = bucket.blob(dest_bucket_pathname)
    if blob.exists():
        raise ValueError(dest_bucket_pathname + ": File already exists in bucket.")
    blob.upload_from_filename(src_volume_filepath)


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """
    Parse a ``gs://`` URI into ``(bucket_name, blob_name)``.
    """
    parsed = urlparse(gcs_uri)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI: {gcs_uri!r}")
    return bucket_name, blob_name


def get_storage_client(client: Optional[storage.Client] = None) -> storage.Client:
    """Return an existing GCS client or create a new one."""
    return client or storage.Client()


def download_gcs_uri_to_filename(
    gcs_uri: str,
    destination_path: str,
    *,
    storage_client: Optional[storage.Client] = None,
) -> str:
    """
    Download a ``gs://`` URI to an explicit local destination path.
    """
    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    client = get_storage_client(storage_client)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(destination_path)
    return destination_path


def download_gcs_uri_to_dir(
    gcs_uri: str,
    destination_dir: str,
    *,
    storage_client: Optional[storage.Client] = None,
) -> str:
    """
    Download a ``gs://`` URI into ``destination_dir`` using the blob basename.
    Returns the full downloaded file path.
    """
    _, blob_name = parse_gcs_uri(gcs_uri)
    destination_path = os.path.join(destination_dir, os.path.basename(blob_name))
    return download_gcs_uri_to_filename(
        gcs_uri,
        destination_path,
        storage_client=storage_client,
    )


def resolve_input_source_to_local_path(
    source: str,
    destination_dir: str,
    *,
    gcp_bucket_name: str = "",
    gcp_blob_prefix: str = "",
    storage_client: Optional[storage.Client] = None,
) -> str:
    """
    Resolve an input source into a local path.

    Supported source formats:
      - absolute path (``/Volumes/...``) or ``dbfs:/...`` path: returned unchanged
      - ``gs://bucket/path``: downloaded to ``destination_dir``
      - object name (e.g. ``foo/bar.csv``): resolved via
        ``gs://{gcp_bucket_name}/{gcp_blob_prefix}/{object_name}``
    """
    value = source.strip()
    if not value:
        raise ValueError("Input source must be non-empty.")

    if value.startswith("/") or value.startswith("dbfs:/"):
        return value

    if value.startswith("gs://"):
        return download_gcs_uri_to_dir(
            value,
            destination_dir,
            storage_client=storage_client,
        )

    if not gcp_bucket_name:
        raise ValueError(
            "gcp_bucket_name is required when source is not a path or gs:// URI."
        )
    if "://" in value:
        raise ValueError(f"Unsupported source URI: {value!r}")

    rel = value.lstrip("/")
    prefix = gcp_blob_prefix.strip().strip("/")
    blob_name = f"{prefix}/{rel}" if prefix else rel
    return download_gcs_uri_to_dir(
        f"gs://{gcp_bucket_name}/{blob_name}",
        destination_dir,
        storage_client=storage_client,
    )


def publish_inference_output_files(
    db_workspace: str,
    institution_name: str,
    external_bucket_name: str,
    sst_job_id: str,
    approved: bool,
) -> str:
    """Publish output files to bucket, with folder determined based on approved parameter,
    and return the status of the job as a string.

    Args:
      db_workspace: The Databricks workspace to get files from.
      institution_name: The Databricks institution name substring used in the schema (e.g. it would be
        'uni_of_datakind' if the gold schema was 'uni_of_datakind_gold') -- this should match the
        "Databricksified" name generated by the webapp.
      external_bucket_name: The destination bucket in GCP.
      sst_job_id: the job run id of this task.
      approved: whether this file should be published in an approved state or not.

    Returns:
      The status string of the job run from Databricks.
    """
    bucket_directory = f"{'approved' if approved else 'unapproved'}/{sst_job_id}"
    volume_path_top_level = f"/Volumes/{db_workspace}/{institution_name}_gold/gold_volume/inference_jobs/{sst_job_id}/ext"
    volume_path_inference_folder = f"{volume_path_top_level}/inference_output"
    dbutils = get_dbutils_or_none()
    if dbutils is None:
        raise RuntimeError(
            "Databricks dbutils are required to publish inference outputs."
        )

    storage_client = storage.Client()
    bucket = storage_client.bucket(external_bucket_name)

    files_to_move = []
    status_string = ""
    for f in dbutils.fs.ls(volume_path_inference_folder):
        filename = f.name
        if filename.startswith("_"):
            if not filename.startswith("_committed") and not filename.startswith(
                "_started"
            ):
                status_string = filename
        elif filename.endswith(".csv"):
            files_to_move.append(f"{volume_path_inference_folder}/{filename}")
    for f in dbutils.fs.ls(volume_path_top_level):
        if f.name.endswith(".png"):
            files_to_move.append(f"{volume_path_top_level}/{f.name}")

    for f in files_to_move:
        new_fname = os.path.basename(f)
        if new_fname.endswith(".csv"):
            new_fname = "inference_output.csv"
        save_file(
            bucket,
            f,
            f"{bucket_directory}/{new_fname}",
        )
    return status_string


def active_gcp_identity() -> str:
    """
    Best-effort string describing the active Google Cloud credentials principal
    (e.g. service account email), or the string 'unknown' if resolution fails.
    """
    try:
        creds, _ = google.auth.default()
        for attr in (
            "service_account_email",
            "service_account_email_address",
            "service_account",
        ):
            if hasattr(creds, attr):
                return str(getattr(creds, attr))
        return str(type(creds))
    except Exception:
        return "unknown"


# --- Validated GCS → local landing directory copy ---


def normalize_gcs_prefix(prefix: str) -> str:
    p = prefix.strip().strip("/")
    return f"{p}/" if p else ""


def _normalize_blob_name(name: str) -> str:
    n = name.strip().lstrip("/").replace("\\", "/")
    for part in n.split("/"):
        if part in ("", ".", ".."):
            raise ValueError(f"Invalid GCS object path: {name!r}")
    return n


def parse_include_blob_paths_json(raw: str) -> list[str]:
    """Parse job/API JSON array of full bucket object paths. Empty / [] = no filter."""
    if raw is None or not str(raw).strip():
        return []
    data = json.loads(str(raw).strip())
    if data == []:
        return []
    if not isinstance(data, list):
        raise ValueError("include_blob_paths_json must be a JSON array of strings")
    out: list[str] = []
    for i, item in enumerate(data):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"include_blob_paths_json entry {i} must be a non-empty string"
            )
        out.append(_normalize_blob_name(item))
    return out


def relative_under_prefix(blob_name: str, gcs_prefix: str) -> str:
    if not blob_name.startswith(gcs_prefix):
        raise ValueError(
            f"Blob {blob_name!r} must start with gcs_source_prefix {gcs_prefix!r} "
            "(pass full object paths as stored in GCS)."
        )
    rel = blob_name[len(gcs_prefix) :].lstrip("/")
    if not rel:
        raise ValueError(f"Blob path equals prefix only (no file): {blob_name!r}")
    return rel


def assert_safe_volume_segment(label: str, value: str) -> str:
    s = value.strip()
    if not s:
        raise ValueError(f"{label} must be non-empty after strip.")
    if not _SAFE_VOLUME_SEGMENT.fullmatch(s):
        raise ValueError(
            f"{label} must match {_SAFE_VOLUME_SEGMENT.pattern!r} "
            f"(alphanumeric start; only . _ - allowed); got {value!r}"
        )
    if ".." in s:
        raise ValueError(f"{label} must not contain '..'; got {value!r}")
    return s


def _assert_safe_blob_relative(relative: str) -> None:
    if not relative or relative.startswith(("/", "\\")):
        raise ValueError(f"Unsafe GCS relative path (absolute or empty): {relative!r}")
    norm = relative.replace("\\", "/")
    for part in norm.split("/"):
        if part in ("", ".", ".."):
            raise ValueError(
                f"Unsafe GCS relative path segment in {relative!r} "
                f"(empty, '.', or '..' not allowed)."
            )


def dest_local_under_landing(landing_dir: str, relative: str) -> str:
    """Return local filesystem path for dest file; ensure parent resolves under landing_dir."""
    _assert_safe_blob_relative(relative)
    dest_path = os.path.join(landing_dir, relative)
    dest_local = local_fs_path(dest_path)
    land_local = local_fs_path(landing_dir)
    dest_parent = os.path.dirname(dest_local)
    if dest_parent:
        os.makedirs(dest_parent, exist_ok=True)
    land_resolved = os.path.realpath(land_local)
    parent_resolved = os.path.realpath(dest_parent if dest_parent else land_local)
    try:
        common = os.path.commonpath([land_resolved, parent_resolved])
    except ValueError as e:
        raise ValueError(
            f"Refusing destination outside landing dir for relative {relative!r}: {e}"
        ) from e
    if common != land_resolved:
        raise ValueError(
            f"Refusing destination outside landing dir: {relative!r} -> {dest_local}"
        )
    return dest_local


def write_success_marker(
    landing_dir: str,
    *,
    copied: int,
    bucket_name: str,
    gcs_prefix: str,
    storage_layout: str,
    copy_mode: str,
    include_blob_paths: Optional[list[str]] = None,
) -> None:
    payload: dict = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "object_count": copied,
        "gcs_uri": f"gs://{bucket_name}/{gcs_prefix}",
        "gcs_prefix": gcs_prefix,
        "marker": SUCCESS_FILENAME,
        "storage_layout": storage_layout,
        "copy_mode": copy_mode,
    }
    if include_blob_paths:
        payload["source_blobs"] = include_blob_paths[:MAX_BLOBS_LISTED_IN_SUCCESS_JSON]
        if len(include_blob_paths) > MAX_BLOBS_LISTED_IN_SUCCESS_JSON:
            payload["source_blobs_truncated"] = True
    marker_local = dest_local_under_landing(landing_dir, SUCCESS_FILENAME)
    with open(marker_local, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info(
        "Wrote completion marker %s", os.path.join(landing_dir, SUCCESS_FILENAME)
    )


def is_transient_gcs_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(exc, gax_exc.GoogleAPICallError):
        code = (
            getattr(exc, "code", None)
            or getattr(exc, "http_status", None)
            or getattr(exc, "grpc_status_code", None)
        )
        if code in HTTP_STATUS_TRANSIENT_GCS:
            return True
    return False


def download_blob_to_file(blob: Blob, dest_local: str) -> None:
    """Download one blob with limited retries for transient GCS errors."""
    delay = BLOB_DOWNLOAD_RETRY_DELAY_INITIAL
    for attempt in range(1, BLOB_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            blob.download_to_filename(dest_local)
            return
        except (NotFound, Forbidden):
            raise
        except (
            OSError,
            ConnectionError,
            TimeoutError,
            gax_exc.GoogleAPICallError,
        ) as e:
            if attempt == BLOB_DOWNLOAD_MAX_ATTEMPTS or not is_transient_gcs_error(e):
                raise
            logging.warning(
                "Transient GCS download error (attempt %s/%s), retrying: %s",
                attempt,
                BLOB_DOWNLOAD_MAX_ATTEMPTS,
                e,
            )
            time.sleep(delay)
            delay = min(delay * 2, BLOB_DOWNLOAD_RETRY_DELAY_CAP)


def copy_validated_blobs_to_landing(
    *,
    bucket_name: str,
    gcs_prefix: str,
    landing_dir: str,
    include_paths: list[str],
    max_objects: int,
    storage_client: Optional[storage.Client] = None,
) -> int:
    """
    Copy selective validated GCS objects into ``landing_dir``.

    Returns the number of objects copied.
    """
    if len(include_paths) > max_objects:
        raise ValueError(
            f"{len(include_paths)} blobs in include list exceeds max_objects ({max_objects})"
        )
    client = get_storage_client(storage_client)
    bucket = client.bucket(bucket_name)
    copied = 0
    for blob_name in include_paths:
        relative = relative_under_prefix(blob_name, gcs_prefix)
        dest_local = dest_local_under_landing(landing_dir, relative)
        blob = bucket.blob(blob_name)
        try:
            download_blob_to_file(blob, dest_local)
        except NotFound as e:
            raise FileNotFoundError(
                "Validated object not found in GCS (check include_blob_paths_json): "
                f"gs://{bucket_name}/{blob_name}"
            ) from e
        copied += 1
        logging.info(
            "Copied gs://%s/%s -> %s",
            bucket_name,
            blob_name,
            dest_local,
        )
    return copied
