"""
Copy objects from an institution GCS prefix (default: validated/) into the
institution's Unity Catalog managed bronze_volume for downstream Databricks work.

Files are **added or overwritten** under ``bronze_volume/<bronze_subdir>/`` (default
``gcs_uploads``). Existing objects in that folder are never deleted by this job.

Optional ``--batch_id`` adds one extra path segment when you want batch-scoped
folders under ``bronze_subdir``.

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
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, Tuple

from google.api_core.exceptions import Forbidden, NotFound
from google.cloud import storage
from google.cloud.storage import Blob

from edvise.utils.gcs import (
    DEFAULT_GCS_PREFIX,
    assert_safe_volume_segment,
    copy_validated_blobs_to_landing,
    dest_local_under_landing,
    download_blob_to_file,
    is_transient_gcs_error,
    normalize_gcs_prefix,
    parse_include_blob_paths_json,
    relative_under_prefix,
    write_success_marker,
)
from edvise.utils.databricks import local_fs_path

Swallowed = Optional[Literal["forbidden", "notfound"]]

# Backward-compatible aliases for tests and callers.
_assert_safe_volume_segment = assert_safe_volume_segment
_is_transient_gcs_error = is_transient_gcs_error
_parse_include_blob_paths_json = parse_include_blob_paths_json
_relative_under_prefix = relative_under_prefix
_dest_local_under_landing = dest_local_under_landing
_write_success_marker = write_success_marker
_download_blob_to_file = download_blob_to_file


@dataclass(frozen=True, slots=True)
class _SyncPaths:
    """Resolved GCS and volume paths for one run."""

    db_workspace: str
    institution: str
    gcs_prefix: str
    batch_id: str
    landing_dir: str
    landing_local: str
    include_paths: list[str]


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


def _build_sync_paths(args: argparse.Namespace) -> _SyncPaths:
    if not args.gcp_bucket_name.strip():
        raise ValueError(
            "gcp_bucket_name is empty; pass the institution bucket when starting the job."
        )

    db_w = assert_safe_volume_segment("DB_workspace", args.DB_workspace.strip())
    inst = assert_safe_volume_segment(
        "databricks_institution_name", args.databricks_institution_name.strip()
    )
    gcs_prefix = normalize_gcs_prefix(args.gcs_source_prefix) or DEFAULT_GCS_PREFIX
    batch_id_raw = (args.batch_id or "").strip()
    batch_id = (
        assert_safe_volume_segment("batch_id", batch_id_raw) if batch_id_raw else ""
    )
    include_paths = parse_include_blob_paths_json(args.include_blob_paths_json)
    bronze_sub = assert_safe_volume_segment("bronze_subdir", args.bronze_subdir)
    bronze_root = f"/Volumes/{db_w}/{inst}_bronze/bronze_volume"
    landing_dir = (
        os.path.join(bronze_root, bronze_sub, batch_id)
        if batch_id
        else os.path.join(bronze_root, bronze_sub)
    )
    landing_local = local_fs_path(landing_dir)
    return _SyncPaths(
        db_workspace=db_w,
        institution=inst,
        gcs_prefix=gcs_prefix,
        batch_id=batch_id,
        landing_dir=landing_dir,
        landing_local=landing_local,
        include_paths=include_paths,
    )


def _run_selective_copies(
    args: argparse.Namespace,
    sp: _SyncPaths,
    client: storage.Client,
) -> int:
    return copy_validated_blobs_to_landing(
        bucket_name=args.gcp_bucket_name,
        gcs_prefix=sp.gcs_prefix,
        landing_dir=sp.landing_dir,
        include_paths=sp.include_paths,
        max_objects=args.max_objects,
        storage_client=client,
    )


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
        dest_local = dest_local_under_landing(sp.landing_dir, relative)
        download_blob_to_file(blob, dest_local)
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
    include_paths = sp.include_paths
    selective = bool(include_paths)

    try:
        if selective:
            copied = _run_selective_copies(args, sp, client)
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

    write_success_marker(
        sp.landing_dir,
        copied=copied,
        bucket_name=args.gcp_bucket_name,
        gcs_prefix=sp.gcs_prefix,
        storage_layout="batch_scoped" if sp.batch_id else "flat",
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
        "--batch_id",
        default="",
        help=(
            "Optional batch UUID path segment under bronze_subdir. "
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
    gcs_prefix = normalize_gcs_prefix(args.gcs_source_prefix) or DEFAULT_GCS_PREFIX
    include_paths = parse_include_blob_paths_json(args.include_blob_paths_json)
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
