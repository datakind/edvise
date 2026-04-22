"""
Copy objects from an institution GCS prefix (default: validated/) into the
institution's Unity Catalog managed bronze_volume for downstream Databricks work.

Intended to run as a single-task Databricks job (see pipelines/pdp/resources).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterator

from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import storage


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
                "raise --max_objects or narrow the prefix."
            )
        yield blob


def sync_validated_to_bronze(args: argparse.Namespace) -> int:
    gcs_prefix = _normalize_gcs_prefix(args.gcs_source_prefix)
    if not gcs_prefix:
        gcs_prefix = "validated/"

    bronze_root = (
        f"/Volumes/{args.DB_workspace}/"
        f"{args.databricks_institution_name}_bronze/bronze_volume"
    )
    landing_dir = os.path.join(
        bronze_root, args.bronze_subdir.strip("/"), args.sync_run_id.strip("/")
    )
    os.makedirs(local_fs_path(landing_dir), exist_ok=True)

    strict = args.strict if args.strict is not None else (not in_databricks())
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
            dest_path = os.path.join(landing_dir, relative)
            dest_parent = os.path.dirname(dest_path)
            if dest_parent:
                os.makedirs(local_fs_path(dest_parent), exist_ok=True)
            blob.download_to_filename(local_fs_path(dest_path))
            copied += 1
            logging.info("Copied gs://%s/%s -> %s", args.gcp_bucket_name, blob.name, dest_path)
    except Forbidden as e:
        msg = (
            f"GCS 403 listing or reading gs://{args.gcp_bucket_name}/{gcs_prefix}: {e}"
        )
        if strict:
            raise
        logging.error("%s (strict=False, exiting with 0 copied)", msg)
        return 0
    except NotFound as e:
        msg = f"GCS bucket not found gs://{args.gcp_bucket_name}: {e}"
        if strict:
            raise
        logging.error("%s (strict=False)", msg)
        return 0

    if copied == 0 and args.require_at_least_one_file:
        raise FileNotFoundError(
            f"No files found under gs://{args.gcp_bucket_name}/{gcs_prefix}"
        )

    logging.info(
        "gcs_validated_to_bronze_sync finished: %s files -> %s",
        copied,
        landing_dir,
    )

    dbutils = get_dbutils()
    if dbutils:
        dbutils.jobs.taskValues.set(key="synced_file_count", value=str(copied))
        dbutils.jobs.taskValues.set(key="bronze_landing_path", value=landing_dir)

    return copied


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
        default=10_000,
        help="Safety cap on number of objects to copy (default: 10000)",
    )
    parser.add_argument(
        "--require_at_least_one_file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if no objects were copied (default: true)",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If false, GCS auth/not-found errors log and exit 0 with no copies. "
        "Default: strict outside Databricks, lenient inside.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = parse_arguments()
    sync_validated_to_bronze(args)


if __name__ == "__main__":
    main()
