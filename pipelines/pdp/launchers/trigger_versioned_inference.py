#!/usr/bin/env python3
"""
Databricks task 3: submit the full PDP inference pipeline from the archived bundle.

Reads ``databricks_bundle_snapshot/resources/github_pdp_inference.yml`` materialized
at ``pipeline_version``, submits a multi-task Jobs API run with ``git_source`` at that
SHA, and polls until the child run finishes (Task 3 fails if child inference fails).
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
from pipelines.pdp.launchers.model_metadata import (  # noqa: E402
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)

LOGGER = logging.getLogger("trigger_versioned_inference")
DEFAULT_RELEASE_BASE = "/Volumes/dev_sst_02/default/edvise_releases"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit versioned PDP inference (multi-task) from archived bundle YAML "
            "at pipeline_version git SHA."
        ),
    )
    parser.add_argument("--databricks_institution_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument(
        "--release_base_path",
        default=DEFAULT_RELEASE_BASE,
    )
    parser.add_argument(
        "--git_url",
        default=DEFAULT_GIT_URL,
        help="Git remote for inference task source (default: datakind/edvise on GitHub).",
    )
    parser.add_argument(
        "--cohort_file_name",
        default="",
        help="Override inference job parameter cohort_file_name.",
    )
    parser.add_argument(
        "--course_file_name",
        default="",
        help="Override inference job parameter course_file_name.",
    )
    parser.add_argument(
        "--gcp_bucket_name",
        default="",
        help="Override inference job parameter gcp_bucket_name.",
    )
    parser.add_argument(
        "--datakind_notification_email",
        default="",
    )
    parser.add_argument(
        "--DK_CC_EMAIL",
        default="",
    )
    parser.add_argument(
        "--ds_run_as",
        default="",
        help="Service principal / run-as for inference tasks (job parameter).",
    )
    parser.add_argument(
        "--service_account_executer",
        default="",
    )
    parser.add_argument(
        "--datakind_group_to_manage_workflow",
        default="",
        help="Group granted CAN_MANAGE_RUN on the submitted inference run.",
    )
    parser.add_argument(
        "--viewer_user",
        default="",
        help="User granted CAN_VIEW on the submitted inference run.",
    )
    parser.add_argument(
        "--inference_output_run_id",
        default="",
        help=(
            "Optional inference output key (maps to job parameter db_run_id). "
            "Use the same value as a previous run to overwrite silver Delta tables and "
            "gold volume inference_jobs paths (e.g. 32-char hex from table names, or "
            "versioned_<hex>). If empty, a new versioned_<uuid> id is generated."
        ),
    )
    parser.add_argument(
        "--inference_parameters_json",
        default="",
        help="Optional JSON object of extra job parameter overrides.",
    )
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


def _optional_arg(value: str) -> str | None:
    s = (value or "").strip()
    return s if s else None


def _build_parameter_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides: dict[str, str] = {
        "databricks_institution_name": args.databricks_institution_name.strip(),
        "model_name": args.model_name.strip(),
        "DB_workspace": args.DB_workspace.strip(),
    }
    for key in (
        "cohort_file_name",
        "course_file_name",
        "gcp_bucket_name",
        "datakind_notification_email",
        "DK_CC_EMAIL",
        "ds_run_as",
        "service_account_executer",
        "datakind_group_to_manage_workflow",
        "viewer_user",
    ):
        val = _optional_arg(getattr(args, key))
        if val is not None:
            overrides[key] = val
    raw = (args.inference_parameters_json or "").strip()
    if raw:
        extra = json.loads(raw)
        if not isinstance(extra, dict):
            msg = "--inference_parameters_json must be a JSON object"
            raise ValueError(msg)
        for k, v in extra.items():
            if v is not None:
                overrides[str(k)] = str(v)
    out_id = _optional_arg(getattr(args, "inference_output_run_id", ""))
    if out_id is not None:
        overrides["db_run_id"] = out_id
    return overrides


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

    try:
        param_overrides = _build_parameter_overrides(args)
    except (json.JSONDecodeError, ValueError) as exc:
        LOGGER.error("Invalid inference parameter overrides: %s", exc)
        return 1

    spark = get_spark_session()
    if spark is None:
        LOGGER.error("SparkSession is required (run on Databricks).")
        return 1

    resolved = resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace=db_ws,
        databricks_institution_name=inst,
        model_name=model,
        logger=LOGGER,
    )
    if resolved is None:
        return 1
    model_run_id, pipeline_version = resolved
    LOGGER.info(
        "Triggering inference for model_run_id=%s pipeline_version=%s",
        model_run_id,
        pipeline_version,
    )

    release_dir = resolve_release_dir(args.release_base_path, pipeline_version)
    if not release_dir.is_dir():
        LOGGER.error("Release bundle not found: %s (run materialize_runtime_bundle)", release_dir)
        return 1

    try:
        run_id = submit_versioned_inference_from_bundle(
            release_dir,
            pipeline_version=pipeline_version,
            parameter_overrides=param_overrides,
            git_url=(args.git_url or DEFAULT_GIT_URL).strip(),
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            poll_interval_seconds=args.poll_interval_seconds,
            logger=LOGGER,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Failed to submit or complete versioned inference: %s", exc)
        return 1

    if not args.dry_run:
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
