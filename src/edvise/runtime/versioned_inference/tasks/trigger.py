"""Task 3: submit versioned inference from archived bundle."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from edvise.runtime.versioned_inference.submit import (
    DEFAULT_GIT_URL,
    submit_versioned_inference_from_bundle,
)
from edvise.runtime.versioned_inference.cli import (
    add_inference_trigger_args,
    build_launcher_trigger_inputs,
    optional_model_run_id,
)
from edvise.runtime.versioned_inference.model_resolution import (
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)
from edvise.runtime.versioned_inference.pipeline_version_ref import git_ref_kind
from edvise.runtime.versioned_inference.dab_layout import resolve_dab_bundle_layout
from edvise.runtime.versioned_inference.run_metadata import (
    record_versioned_inference_launcher_event,
    resolve_launcher_run_id,
)

LOGGER = logging.getLogger("trigger_versioned_inference")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit versioned PDP inference (multi-task) from archived bundle YAML "
            "at pipeline_version (Git SHA or release tag)."
        ),
    )
    add_inference_trigger_args(parser)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit inference and exit without polling for child run completion.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=30.0,
        help="Seconds between child run status polls (default: 30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log submit payload without calling the Jobs API.",
    )
    return parser.parse_args(argv)


def _fail(
    *,
    catalog: str,
    inst: str,
    model: str,
    model_run_id: str | None,
    archived_pipeline_version: str | None,
    launcher_run_id: str | None,
    message: str,
) -> int:
    LOGGER.error("%s", message)
    record_versioned_inference_launcher_event(
        catalog=catalog,
        event="failed",
        databricks_institution_name=inst,
        model_name=model,
        model_run_id=model_run_id,
        archived_pipeline_version=archived_pipeline_version,
        launcher_run_id=launcher_run_id,
        error_message=message,
        payload={"task": "trigger_versioned_inference"},
        logger=LOGGER,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
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
        LOGGER.error(
            "Require --databricks_institution_name, --model_name, and --DB_workspace."
        )
        return 1

    launcher_run_id = resolve_launcher_run_id(getattr(args, "launcher_run_id", ""))
    if not launcher_run_id:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=None,
            archived_pipeline_version=None,
            launcher_run_id=None,
            message=(
                "launcher_run_id is required (job parameter launcher_run_id with "
                "default {{job.run_id}})."
            ),
        )
    try:
        inputs = build_launcher_trigger_inputs(args, default_git_url=DEFAULT_GIT_URL)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=None,
            archived_pipeline_version=None,
            launcher_run_id=launcher_run_id,
            message=f"Invalid inference parameter overrides: {exc}",
        )

    spark = get_spark_session()
    if spark is None:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=None,
            archived_pipeline_version=None,
            launcher_run_id=launcher_run_id,
            message="SparkSession is required (run on Databricks).",
        )

    resolved = resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace=db_ws,
        databricks_institution_name=inst,
        model_name=model,
        model_run_id_override=optional_model_run_id(args),
        logger=LOGGER,
    )
    if resolved is None:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=None,
            archived_pipeline_version=None,
            launcher_run_id=launcher_run_id,
            message="Could not resolve model_run_id / pipeline_version",
        )
    model_run_id, archived_pipeline_version = resolved
    layout = resolve_dab_bundle_layout(inputs.schema_type)
    LOGGER.info(
        "Triggering inference for model_run_id=%s archived_pipeline_version=%s (git %s)",
        model_run_id,
        archived_pipeline_version,
        git_ref_kind(archived_pipeline_version),
    )

    release_dir = resolve_release_dir(inputs.release_base_path, archived_pipeline_version)
    if not release_dir.is_dir():
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            message=(
                f"Release bundle not found: {release_dir} "
                "(run materialize_runtime_bundle first)"
            ),
        )

    try:
        run_id = submit_versioned_inference_from_bundle(
            release_dir,
            archived_pipeline_version=archived_pipeline_version,
            parameter_overrides=inputs.param_overrides,
            extra_parameter_overrides=inputs.extra_param_overrides,
            stable_trigger=inputs.stable_trigger,
            git_url=inputs.git_url,
            inference_job_key=layout.inference_job_key,
            inference_yml_relative=layout.inference_yml_snapshot_rel,
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            poll_interval_seconds=args.poll_interval_seconds,
            logger=LOGGER,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Failed to submit or complete versioned inference: {exc}",
        )

    if not args.dry_run:
        db_run_id = inputs.param_overrides.get("db_run_id") or launcher_run_id
        if inputs.extra_param_overrides.get("db_run_id"):
            db_run_id = inputs.extra_param_overrides["db_run_id"]
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="completed" if not args.no_wait else "started",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            child_inference_run_id=run_id,
            cohort_dataset_name=inputs.param_overrides.get("cohort_file_name"),
            course_dataset_name=inputs.param_overrides.get("course_file_name"),
            payload={
                "task": "trigger_versioned_inference",
                "no_wait": args.no_wait,
                "parent_launcher_run_id": launcher_run_id,
                "child_inference_run_id": str(run_id),
                "db_run_id": db_run_id,
            },
            logger=LOGGER,
        )
        if args.no_wait:
            LOGGER.info(
                "Versioned inference submitted (no-wait; training model_run_id=%s, "
                "parent_launcher_run_id=%s, child_inference_run_id=%s, db_run_id=%s)",
                model_run_id,
                launcher_run_id,
                run_id,
                db_run_id,
            )
        else:
            LOGGER.info(
                "Versioned inference completed successfully (training model_run_id=%s, "
                "parent_launcher_run_id=%s, child_inference_run_id=%s, db_run_id=%s)",
                model_run_id,
                launcher_run_id,
                run_id,
                db_run_id,
            )
    return 0
