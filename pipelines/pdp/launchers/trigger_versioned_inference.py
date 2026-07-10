#!/usr/bin/env python3
"""
Databricks task 3: submit the full PDP inference pipeline from the archived bundle.

Reads materialized YAML at ``pipeline_version`` (Git SHA in dev, release tag in staging),
submits a multi-task Jobs API run pinned to that ref, and polls until completion.
Fails closed — no fallback to current-deploy inference.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path


def _setup_import_path() -> None:
    try:
        launcher = Path(__file__).resolve().parent
    except NameError:
        launcher = None
        for frame_info in inspect.stack():
            fn = frame_info.filename
            if not fn or fn.startswith("<"):
                continue
            path = Path(fn).resolve()
            if path.parent.name == "launchers" and path.parent.parent.name == "pdp":
                launcher = path.parent
                break
        if launcher is None:
            candidate = Path.cwd() / "pipelines" / "pdp" / "launchers"
            if candidate.is_dir():
                launcher = candidate.resolve()
        if launcher is None:
            msg = "Cannot locate pipelines/pdp/launchers (Databricks import bootstrap)"
            raise RuntimeError(msg)
    launcher_str = str(launcher.resolve())
    if launcher_str not in sys.path:
        sys.path.insert(0, launcher_str)
    from _paths import ensure_repo_root_on_sys_path  # noqa: WPS433

    ensure_repo_root_on_sys_path()


_setup_import_path()

from pipelines.pdp.launchers.inference_job_submit import (  # noqa: E402
    DEFAULT_GIT_URL,
    submit_versioned_inference_from_bundle,
)
from pipelines.pdp.launchers.launcher_cli import (  # noqa: E402
    add_inference_trigger_args,
    build_launcher_trigger_inputs,
    optional_model_run_id,
)
from pipelines.pdp.launchers.launcher_run_metadata import (  # noqa: E402
    get_databricks_run_id,
    record_versioned_inference_launcher_event,
)
from pipelines.pdp.launchers.model_metadata import (  # noqa: E402
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)
from pipelines.pdp.launchers.pipeline_version_ref import git_ref_kind  # noqa: E402

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
    pipeline_version: str | None,
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
        pipeline_version=pipeline_version,
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

    launcher_run_id = get_databricks_run_id()
    try:
        inputs = build_launcher_trigger_inputs(args, default_git_url=DEFAULT_GIT_URL)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=None,
            pipeline_version=None,
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
            pipeline_version=None,
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
            pipeline_version=None,
            launcher_run_id=launcher_run_id,
            message="Could not resolve model_run_id / pipeline_version",
        )
    model_run_id, pipeline_version = resolved
    LOGGER.info(
        "Triggering inference for model_run_id=%s pipeline_version=%s (git %s)",
        model_run_id,
        pipeline_version,
        git_ref_kind(pipeline_version),
    )

    release_dir = resolve_release_dir(inputs.release_base_path, pipeline_version)
    if not release_dir.is_dir():
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            message=(
                f"Release bundle not found: {release_dir} "
                "(run materialize_runtime_bundle first)"
            ),
        )

    try:
        run_id = submit_versioned_inference_from_bundle(
            release_dir,
            pipeline_version=pipeline_version,
            parameter_overrides=inputs.param_overrides,
            extra_parameter_overrides=inputs.extra_param_overrides,
            stable_trigger=inputs.stable_trigger,
            git_url=inputs.git_url,
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
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Failed to submit or complete versioned inference: {exc}",
        )

    if not args.dry_run:
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="completed" if not args.no_wait else "started",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            child_inference_run_id=run_id,
            cohort_dataset_name=inputs.param_overrides.get("cohort_file_name"),
            course_dataset_name=inputs.param_overrides.get("course_file_name"),
            payload={"task": "trigger_versioned_inference", "no_wait": args.no_wait},
            logger=LOGGER,
        )
        if args.no_wait:
            LOGGER.info(
                "Versioned inference submitted (no-wait; training model_run_id=%s, "
                "jobs run_id=%s)",
                model_run_id,
                run_id,
            )
        else:
            LOGGER.info(
                "Versioned inference completed successfully (training model_run_id=%s, "
                "jobs run_id=%s)",
                model_run_id,
                run_id,
            )
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
