"""
Copy objects from an institution GCS prefix (default: validated/) into the
institution's Unity Catalog managed bronze_volume for downstream Databricks work.

Files are **added or overwritten** under ``bronze_volume/<bronze_subdir>/`` (default
``gcs_uploads``). Existing objects in that folder are never deleted by this job.

Optional ``--sync_run_id`` adds one extra path segment when you want run-scoped
folders.

Optional ``--include_blob_paths_json`` (JSON array of full GCS object paths, e.g.
``["validated/foo.csv"]``) copies **only** those objects from the current validation
batch. When non-empty, prefix listing is skipped. When empty (``[]``), all objects
under ``gcs_source_prefix`` are copied (same additive behavior; no deletes).

``_SUCCESS.json`` is written when the copy loop finishes without the early
``no files + require_at_least_one`` exit. That includes 0 files copied with
``--require_at_least_one_file`` false. With ``--require_at_least_one_file`` true
and 0 files, the run exits before writing the marker; ``main`` then raises.

Intended to run as a single-task Databricks job (see pipelines/ingestion/shared/resources).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Literal, Optional, Tuple

from google.api_core import exceptions as gax_exc
from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import storage
from google.cloud.storage import Blob

# Single path segment for landing subdirs (no traversal, no nested path in param).
_SAFE_VOLUME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

SUCCESS_FILENAME = "_SUCCESS.json"
DEFAULT_GCS_PREFIX = "validated/"
MAX_BLOBS_LISTED_IN_SUCCESS_JSON = 50
BLOB_DOWNLOAD_MAX_ATTEMPTS = 3
BLOB_DOWNLOAD_RETRY_DELAY_INITIAL = 0.5
BLOB_DOWNLOAD_RETRY_DELAY_CAP = 4.0
# HTTP codes where GCS/transport often benefits from a retry
HTTP_STATUS_TRANSIENT_GCS: Tuple[int, ...] = (429, 500, 502, 503, 504)

Swallowed = Optional[Literal["forbidden", "notfound"]]


@dataclass(frozen=True, slots=True)
class _SyncPaths:
    """Resolved GCS and volume paths for one run."""

    db_workspace: str
    institution: str
    gcs_prefix: str
    sync_id: str
    landing_dir: str
    landing_local: str
    include_paths: list[str]


def local_fs_path(path: str) -> str:
    """Map dbfs: URI to a local /dbfs path; otherwise return the path unchanged."""
    return (
        path.replace("dbfs:/", "/dbfs/") if path and path.startswith("dbfs:/") else path
    )


def in_databricks() -> bool:
    """True if running in a Databricks/Spark driver environment."""
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER"))


def get_dbutils() -> Any | None:
    """
    Return Databricks ``dbutils`` in-cluster, or None when not available
    (e.g. local pytest).
    """
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore[import-not-found]

        return dbutils
    except (ImportError, ModuleNotFoundError):
        return None


def _normalize_gcs_prefix(prefix: str) -> str:
    p = prefix.strip().strip("/")
    return f"{p}/" if p else ""


def _normalize_blob_name(name: str) -> str:
    n = name.strip().lstrip("/").replace("\\", "/")
    for part in n.split("/"):
        if part in ("", ".", ".."):
            raise ValueError(f"Invalid GCS object path: {name!r}")
    return n


def _parse_include_blob_paths_json(raw: str) -> list[str]:
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


def _relative_under_prefix(blob_name: str, gcs_prefix: str) -> str:
    if not blob_name.startswith(gcs_prefix):
        raise ValueError(
            f"Blob {blob_name!r} must start with gcs_source_prefix {gcs_prefix!r} "
            "(pass full object paths as stored in GCS)."
        )
    rel = blob_name[len(gcs_prefix) :].lstrip("/")
    if not rel:
        raise ValueError(f"Blob path equals prefix only (no file): {blob_name!r}")
    return rel


def _assert_safe_volume_segment(label: str, value: str) -> str:
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
    """Reject object keys that could escape the landing directory via path segments."""
    if not relative or relative.startswith(("/", "\\")):
        raise ValueError(f"Unsafe GCS relative path (absolute or empty): {relative!r}")
    norm = relative.replace("\\", "/")
    for part in norm.split("/"):
        if part in ("", ".", ".."):
            raise ValueError(
                f"Unsafe GCS relative path segment in {relative!r} "
                f"(empty, '.', or '..' not allowed)."
            )


def _dest_local_under_landing(landing_dir: str, relative: str) -> str:
    """
    Return local filesystem path for dest file; ensure parent resolves under landing_dir.
    Creates parent directories as needed.
    """
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


def _write_success_marker(
    landing_dir: str,
    *,
    copied: int,
    bucket_name: str,
    gcs_prefix: str,
    storage_layout: str,
    copy_mode: str,
    include_blob_paths: Optional[list[str]] = None,
) -> None:
    """Write ``_SUCCESS.json`` under ``landing_dir`` (``copied`` may be 0 if the run allowed it)."""
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
    marker_path = os.path.join(landing_dir, SUCCESS_FILENAME)
    marker_local = _dest_local_under_landing(landing_dir, SUCCESS_FILENAME)
    with open(marker_local, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote completion marker %s", marker_path)


def _iter_blobs(
    client: storage.Client, bucket_name: str, prefix: str, max_objects: int
) -> Iterator[Blob]:
    count = 0
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        count += 1
        if count > max_objects:
            raise ValueError(
                f"More than {max_objects} objects under gs://{bucket_name}/{prefix}; "
                "increase --max_objects or narrow the prefix."
            )
        yield blob


def _resolve_strict_mode(strict_mode: str) -> bool:
    if strict_mode == "auto":
        return not in_databricks()
    if strict_mode == "true":
        return True
    if strict_mode == "false":
        return False
    raise ValueError(f"Invalid strict_mode: {strict_mode!r}")


def _is_transient_gcs_error(exc: Exception) -> bool:
    """True for likely-transient GCS/transport failures worth retrying."""
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


def _download_blob_to_file(blob: Blob, dest_local: str) -> None:
    """Download one blob to dest with limited retries for transient GCS errors."""
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
            if attempt == BLOB_DOWNLOAD_MAX_ATTEMPTS or not _is_transient_gcs_error(e):
                raise
            logging.warning(
                "Transient GCS download error (attempt %s/%s), retrying: %s",
                attempt,
                BLOB_DOWNLOAD_MAX_ATTEMPTS,
                e,
            )
            time.sleep(delay)
            delay = min(delay * 2, BLOB_DOWNLOAD_RETRY_DELAY_CAP)


def _build_sync_paths(args: argparse.Namespace) -> _SyncPaths:
    if not args.gcp_bucket_name.strip():
        raise ValueError(
            "gcp_bucket_name is empty; pass the institution bucket when starting the job."
        )

    db_w = _assert_safe_volume_segment("DB_workspace", args.DB_workspace.strip())
    inst = _assert_safe_volume_segment(
        "databricks_institution_name", args.databricks_institution_name.strip()
    )
    gcs_prefix = _normalize_gcs_prefix(args.gcs_source_prefix)
    if not gcs_prefix:
        gcs_prefix = DEFAULT_GCS_PREFIX
    sync_id_raw = (args.sync_run_id or "").strip()
    if sync_id_raw:
        sync_id = _assert_safe_volume_segment("sync_run_id", sync_id_raw)
    else:
        sync_id = ""
    include_paths = _parse_include_blob_paths_json(args.include_blob_paths_json)
    bronze_sub = _assert_safe_volume_segment("bronze_subdir", args.bronze_subdir)
    bronze_root = f"/Volumes/{db_w}/{inst}_bronze/bronze_volume"
    landing_dir = (
        os.path.join(bronze_root, bronze_sub, sync_id)
        if sync_id
        else os.path.join(bronze_root, bronze_sub)
    )
    landing_local = local_fs_path(landing_dir)
    return _SyncPaths(
        db_workspace=db_w,
        institution=inst,
        gcs_prefix=gcs_prefix,
        sync_id=sync_id,
        landing_dir=landing_dir,
        landing_local=landing_local,
        include_paths=include_paths,
    )


def _run_selective_copies(
    args: argparse.Namespace,
    sp: _SyncPaths,
    bucket: storage.Bucket,
) -> int:
    if len(sp.include_paths) > args.max_objects:
        raise ValueError(
            f"{len(sp.include_paths)} blobs in include_blob_paths_json exceeds "
            f"--max_objects ({args.max_objects})"
        )
    copied = 0
    for blob_name in sp.include_paths:
        relative = _relative_under_prefix(blob_name, sp.gcs_prefix)
        dest_local = _dest_local_under_landing(sp.landing_dir, relative)
        blob = bucket.blob(blob_name)
        try:
            _download_blob_to_file(blob, dest_local)
        except NotFound as e:
            raise FileNotFoundError(
                "Validated object not found in GCS (check include_blob_paths_json): "
                f"gs://{args.gcp_bucket_name}/{blob_name}"
            ) from e
        copied += 1
        logging.info(
            "Copied gs://%s/%s -> %s",
            args.gcp_bucket_name,
            blob_name,
            dest_local,
        )
    return copied


def _run_listing_copies(
    args: argparse.Namespace, sp: _SyncPaths, client: storage.Client
) -> int:
    copied = 0
    for blob in _iter_blobs(
        client, args.gcp_bucket_name, sp.gcs_prefix, args.max_objects
    ):
        if not blob.name.startswith(sp.gcs_prefix):
            continue
        relative = blob.name[len(sp.gcs_prefix) :].lstrip("/")
        if not relative:
            continue
        dest_local = _dest_local_under_landing(sp.landing_dir, relative)
        _download_blob_to_file(blob, dest_local)
        copied += 1
        logging.info(
            "Copied gs://%s/%s -> %s",
            args.gcp_bucket_name,
            blob.name,
            dest_local,
        )
    return copied


def _try_set_job_task_values(landing_dir: str, copied: int) -> None:
    """Set multi-task job values; failures are logged only (DRuntime varies)."""
    dbc = get_dbutils()
    if not dbc:
        return
    try:
        dbc.jobs.taskValues.set(key="synced_file_count", value=str(copied))
        dbc.jobs.taskValues.set(key="bronze_landing_path", value=landing_dir)
    except (AttributeError, OSError, RuntimeError, TypeError) as e:
        logging.error(
            "dbutils.jobs.taskValues.set failed; sync finished but job taskValues "
            "unavailable: %s",
            e,
            exc_info=True,
        )


def sync_validated_to_bronze(args: argparse.Namespace) -> Tuple[int, Swallowed]:
    """
    Copy validated GCS objects to the institution bronze volume.

    Returns (copied_count, swallowed_error). ``swallowed_error`` is set when
    ``strict_mode`` is false and GCS returned 403/404 so the caller can surface
    :class:`PermissionError` or bucket-missing instead of "empty prefix".
    """
    sp = _build_sync_paths(args)
    os.makedirs(sp.landing_local, exist_ok=True)
    strict = _resolve_strict_mode(args.strict_mode)
    client = storage.Client()
    bucket = client.bucket(args.gcp_bucket_name)
    include_paths = sp.include_paths
    selective = bool(include_paths)

    try:
        if selective:
            copied = _run_selective_copies(args, sp, bucket)
        else:
            copied = _run_listing_copies(args, sp, client)
    except Forbidden as e:
        msg = f"GCS 403 listing or reading gs://{args.gcp_bucket_name}/{sp.gcs_prefix}: {e}"
        if strict:
            raise
        logging.error("%s (strict_mode=false, exiting with 0 copied)", msg)
        return 0, "forbidden"
    except NotFound as e:
        msg = f"GCS bucket not found gs://{args.gcp_bucket_name}: {e}"
        if strict:
            raise
        logging.error("%s (strict_mode=false)", msg)
        return 0, "notfound"

    if copied == 0 and args.require_at_least_one_file:
        return 0, None

    logging.info(
        "gcs_validated_to_bronze_sync finished: %s files -> %s", copied, sp.landing_dir
    )

    _write_success_marker(
        sp.landing_dir,
        copied=copied,
        bucket_name=args.gcp_bucket_name,
        gcs_prefix=sp.gcs_prefix,
        storage_layout="run_scoped" if sp.sync_id else "flat",
        copy_mode="selective" if selective else "all_under_prefix",
        include_blob_paths=include_paths if selective else None,
    )

    _try_set_job_task_values(sp.landing_dir, copied)

    return copied, None


def _add_cli_path_arguments(parser: argparse.ArgumentParser) -> None:
    """Register volume and GCS path related CLI args."""
    parser.add_argument(
        "--DB_workspace", required=True, help="Unity Catalog catalog name"
    )
    parser.add_argument(
        "--databricks_institution_name",
        required=True,
        help="Databricks institution slug (schema suffix)",
    )
    parser.add_argument(
        "--gcp_bucket_name",
        required=True,
        help="Institution GCS bucket containing validated uploads",
    )
    parser.add_argument(
        "--sync_run_id",
        default="",
        help=(
            "Optional single path segment under bronze_subdir (e.g. job run id). "
            "Leave empty (default) to land files directly under bronze_subdir."
        ),
    )
    parser.add_argument(
        "--gcs_source_prefix",
        default="validated/",
        help="Prefix inside the bucket to copy (default: validated/)",
    )
    parser.add_argument(
        "--bronze_subdir",
        default="gcs_uploads",
        help="Directory under bronze_volume for uploaded copies (default: gcs_uploads)",
    )
    parser.add_argument(
        "--include_blob_paths_json",
        default="[]",
        help=(
            'JSON array of full GCS object paths to copy, e.g. ["validated/a.csv"]. '
            "Each path must start with gcs_source_prefix. Use [] (default) to copy all "
            "objects under the prefix (still additive; no deletes)."
        ),
    )


def _add_cli_mode_arguments(parser: argparse.ArgumentParser) -> None:
    """Register cap and behavior CLI args."""
    parser.add_argument(
        "--max_objects",
        type=int,
        default=1_000,
        help="Safety cap on number of objects to copy (default: 1000)",
    )
    parser.add_argument(
        "--require_at_least_one_file",
        choices=("true", "false"),
        default="true",
        help="If true, fail when zero objects were copied (default: true)",
    )
    parser.add_argument(
        "--strict_mode",
        choices=("auto", "true", "false"),
        default="auto",
        help="auto: strict off-cluster, lenient on DBR (matches PDP pattern). "
        "true/false: force.",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI for the Databricks job (see bundle ``job.parameters``)."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy GCS objects under a prefix into the institution bronze_volume "
            "(additive: overwrite matching paths only; never delete existing files)."
        )
    )
    _add_cli_path_arguments(parser)
    _add_cli_mode_arguments(parser)
    return parser.parse_args()


def _raise_on_zero_copies(
    args: argparse.Namespace, swallowed: Swallowed, copied: int
) -> None:
    if copied != 0 or not args.require_at_least_one_file:
        return
    if swallowed == "forbidden":
        raise PermissionError(
            "GCS returned 403 Forbidden for this identity; "
            "zero objects copied. This is not an empty prefix — fix IAM / bucket access."
        )
    if swallowed == "notfound":
        raise FileNotFoundError(
            "GCS bucket was not found for the given gcp_bucket_name; "
            "zero objects copied."
        )
    gcs_prefix = _normalize_gcs_prefix(args.gcs_source_prefix) or DEFAULT_GCS_PREFIX
    include_paths = _parse_include_blob_paths_json(args.include_blob_paths_json)
    if include_paths:
        raise FileNotFoundError(
            "No objects were copied from include_blob_paths_json "
            f"(check paths exist under gs://{args.gcp_bucket_name}/): {include_paths!r}"
        )
    raise FileNotFoundError(
        f"No files found under gs://{args.gcp_bucket_name}/{gcs_prefix}"
    )


def main() -> None:
    """Entry point for the driver script."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = parse_arguments()
    args.require_at_least_one_file = args.require_at_least_one_file == "true"

    copied, swallowed = sync_validated_to_bronze(args)
    _raise_on_zero_copies(args, swallowed, copied)


if __name__ == "__main__":
    main()
