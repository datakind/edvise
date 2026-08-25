"""Task 1: resolve archived pipeline_version and materialize runtime bundle."""

from __future__ import annotations

import argparse
import logging
import sys

from edvise.runtime.versioned_inference.bundle.from_dab import inference_yml_path
from edvise.runtime.versioned_inference.bundle.materialize import (
    DEFAULT_GITHUB_REPO,
    materialize_runtime_bundle_dir,
)
from edvise.runtime.versioned_inference.cli import (
    add_model_resolution_args,
    launcher_schema_type,
    optional_model_run_id,
)
from edvise.runtime.versioned_inference.dab_layout import resolve_dab_bundle_layout
from edvise.runtime.versioned_inference.model_resolution import (
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)
from edvise.runtime.versioned_inference.pipeline_version_ref import git_ref_kind
from edvise.runtime.versioned_inference.release_config import resolve_release_base_path
from edvise.runtime.versioned_inference.run_metadata import (
    record_launcher_failures,
    record_versioned_inference_launcher_event,
    resolve_launcher_run_id,
)

LOGGER = logging.getLogger("materialize_runtime_bundle")
TASK_NAME = "materialize_runtime_bundle"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve pipeline_version and materialize DAB YAML snapshot on the release volume."
        ),
    )
    add_model_resolution_args(parser)
    parser.add_argument(
        "--github_repo",
        default=DEFAULT_GITHUB_REPO,
        help="GitHub org/repo for raw YAML fetch (default: datakind/edvise).",
    )
    parser.add_argument(
        "--skip-snapshot-if-present",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip GitHub fetch when inference YAML snapshot already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    inst = args.databricks_institution_name.strip()
    model = args.model_name.strip()
    db_ws = args.DB_workspace.strip()
    if not inst or not model or not db_ws:
        raise ValueError(
            "Require --databricks_institution_name, --model_name, and --DB_workspace."
        )

    launcher_run_id = resolve_launcher_run_id(getattr(args, "launcher_run_id", ""))
    record_versioned_inference_launcher_event(
        catalog=db_ws,
        event="started",
        databricks_institution_name=inst,
        model_name=model,
        launcher_run_id=launcher_run_id,
        logger=LOGGER,
    )

    with record_launcher_failures(
        catalog=db_ws,
        databricks_institution_name=inst,
        model_name=model,
        launcher_run_id=launcher_run_id,
        task=TASK_NAME,
        logger=LOGGER,
    ) as event:
        spark = get_spark_session()
        if spark is None:
            raise RuntimeError("SparkSession is required (run on Databricks).")

        resolved = resolve_model_run_and_pipeline_version(
            spark=spark,
            db_workspace=db_ws,
            databricks_institution_name=inst,
            model_name=model,
            model_run_id_override=optional_model_run_id(args),
            logger=LOGGER,
        )
        if resolved is None:
            raise ValueError("Could not resolve model_run_id / pipeline_version")
        model_run_id, archived_pipeline_version = resolved
        event.model_run_id = model_run_id
        event.archived_pipeline_version = archived_pipeline_version
        LOGGER.info(
            "Materializing bundle for model_run_id=%s archived_pipeline_version=%s (git %s)",
            model_run_id,
            archived_pipeline_version,
            git_ref_kind(archived_pipeline_version),
        )

        release_base = resolve_release_base_path(db_ws, args.release_base_path)
        release_dir = resolve_release_dir(release_base, archived_pipeline_version)
        schema_type = launcher_schema_type(args)
        layout = resolve_dab_bundle_layout(schema_type)
        materialize_runtime_bundle_dir(
            release_dir,
            archived_pipeline_version,
            schema_type=schema_type,
            git_ref=archived_pipeline_version,
            github_repo=args.github_repo.strip() or DEFAULT_GITHUB_REPO,
            skip_snapshot_if_present=args.skip_snapshot_if_present,
            logger=LOGGER,
        )

        marker = inference_yml_path(release_dir, layout.inference_yml_snapshot_rel)
        if not marker.is_file():
            raise FileNotFoundError(f"DAB snapshot missing after materialize: {marker}")

        LOGGER.info("Runtime bundle materialized at %s", release_dir)
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="completed",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            payload={
                "bundle_materialized": str(release_dir),
                "task": TASK_NAME,
                "schema_type": layout.schema_type,
            },
            logger=LOGGER,
        )
