"""
Copy objects from an institution GCS prefix (default: validated/) into the
institution's Unity Catalog managed bronze_volume for downstream Databricks work.

Intended to run as a single-task Databricks job (see pipelines/pdp/resources).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Iterator, Literal, Optional, Tuple

from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import storage

# Single path segment for landing subdirs (no traversal, no nested path in param).
_SAFE_VOLUME_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")

SUCCESS_FILENAME = "_SUCCESS.json"


def local_fs_path(p: str) -> str:
    return p.replace("dbfs:/", "/dbfs/") if p and p.startswith("dbfs:/") else p


def in_databricks() -> bool:
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER"))


def get_dbutils():
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore

        return dbutils
    except Exception:
        return None


def _normalize_gcs_prefix(prefix: str) -> str:
    p = prefix.strip().strip("/")
    return f"{p}/" if p else ""


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
) -> None:
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "object_count": copied,
        "gcs_uri": f"gs://{bucket_name}/{gcs_prefix}",
        "gcs_prefix": gcs_prefix,
        "marker": SUCCESS_FILENAME,
    }
    marker_path = os.path.join(landing_dir, SUCCESS_FILENAME)
    marker_local = _dest_local_under_landing(landing_dir, SUCCESS_FILENAME)
    with open(marker_local, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info("Wrote completion marker %s", marker_path)


def _iter_blobs(
    client: storage.Client, bucket_name: str, prefix: str, max_objects: int
) -> Iterator[storage.Blob]:
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


Swallowed = Optional[Literal["forbidden", "notfound"]]


def sync_validated_to_bronze(args: argparse.Namespace) -> Tuple[int, Swallowed]:
    """
    Returns (copied_count, swallowed_error).
    swallowed_error is set when strict_mode=false and GCS returned 403/404 so the caller
    can surface PermissionError / bucket missing instead of 'empty prefix'.
    """
    if not args.gcp_bucket_name.strip():
        raise ValueError(
            "gcp_bucket_name is empty; pass the institution bucket when starting the job."
        )

    gcs_prefix = _normalize_gcs_prefix(args.gcs_source_prefix)
    if not gcs_prefix:
        gcs_prefix = "validated/"

    bronze_sub = _assert_safe_volume_segment("bronze_subdir", args.bronze_subdir)
    sync_id = _assert_safe_volume_segment("sync_run_id", args.sync_run_id)

    bronze_root = (
        f"/Volumes/{args.DB_workspace}/"
        f"{args.databricks_institution_name}_bronze/bronze_volume"
    )
    landing_dir = os.path.join(bronze_root, bronze_sub, sync_id)
    os.makedirs(local_fs_path(landing_dir), exist_ok=True)

    strict = _resolve_strict_mode(args.strict_mode)
    client = storage.Client()

    copied = 0
    try:
        for blob in _iter_blobs(
            client, args.gcp_bucket_name, gcs_prefix, args.max_objects
        ):
            if not blob.name.startswith(gcs_prefix):
                continue
            relative = blob.name[len(gcs_prefix) :].lstrip("/")
            if not relative:
                continue
            dest_local = _dest_local_under_landing(landing_dir, relative)
            blob.download_to_filename(dest_local)
            copied += 1
            logging.info(
                "Copied gs://%s/%s -> %s", args.gcp_bucket_name, blob.name, dest_local
            )
    except Forbidden as e:
        msg = (
            f"GCS 403 listing or reading gs://{args.gcp_bucket_name}/{gcs_prefix}: {e}"
        )
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
        "gcs_validated_to_bronze_sync finished: %s files -> %s",
        copied,
        landing_dir,
    )

    _write_success_marker(
        landing_dir,
        copied=copied,
        bucket_name=args.gcp_bucket_name,
        gcs_prefix=gcs_prefix,
    )

    dbutils = get_dbutils()
    if dbutils:
        dbutils.jobs.taskValues.set(key="synced_file_count", value=str(copied))
        dbutils.jobs.taskValues.set(key="bronze_landing_path", value=landing_dir)

    return copied, None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy GCS objects under a prefix into the institution bronze_volume "
            "(default prefix: validated/)."
        )
    )
    parser.add_argument("--DB_workspace", required=True, help="Unity Catalog catalog name")
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
        required=True,
        help="Subfolder under bronze landing (e.g. batch id or Databricks job run id)",
    )
    parser.add_argument(
        "--gcs_source_prefix",
        default="validated/",
        help="Prefix inside the bucket to copy (default: validated/)",
    )
    parser.add_argument(
        "--bronze_subdir",
        default="gcs_validated_sync",
        help="Directory under bronze_volume before sync_run_id (default: gcs_validated_sync)",
    )
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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = parse_arguments()
    args.require_at_least_one_file = args.require_at_least_one_file == "true"

    copied, swallowed = sync_validated_to_bronze(args)

    if copied == 0 and args.require_at_least_one_file:
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
        gcs_prefix = _normalize_gcs_prefix(args.gcs_source_prefix) or "validated/"
        raise FileNotFoundError(
            f"No files found under gs://{args.gcp_bucket_name}/{gcs_prefix}"
        )


if __name__ == "__main__":
    main()
