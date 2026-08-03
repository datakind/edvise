"""Task 2: validate archived bundle and parameter contract."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from edvise.runtime.versioned_inference.bundle.from_dab import (
    build_effective_release,
    inference_yml_path,
    load_inference_job_definition,
)
from edvise.runtime.versioned_inference.dab_layout import resolve_dab_bundle_layout
from edvise.runtime.versioned_inference.parameters import (
    resolve_versioned_job_parameters,
)
from edvise.runtime.versioned_inference.submit import DEFAULT_GIT_URL
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
from edvise.runtime.versioned_inference.run_metadata import (
    record_versioned_inference_launcher_event,
    resolve_launcher_run_id,
)
from edvise.runtime.versioned_inference.runtime_compat import check_runtime_bundle_compatibility

LOGGER = logging.getLogger("versioned_inference_launcher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate archived inference bundle, cluster compatibility, and parameter contract."
        ),
    )
    add_inference_trigger_args(parser)
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
        payload={"task": "versioned_inference_launcher_validate"},
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

    release_dir = resolve_release_dir(inputs.release_base_path, archived_pipeline_version)
    LOGGER.info(
        "Release bundle directory: %s (archived_pipeline_version=%s, git %s)",
        release_dir,
        archived_pipeline_version,
        git_ref_kind(archived_pipeline_version),
    )
    if not release_dir.is_dir():
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Release bundle directory not found: {release_dir}",
        )

    try:
        effective = build_effective_release(
            release_dir,
            archived_pipeline_version,
            inference_yml_relative=layout.inference_yml_snapshot_rel,
            inference_job_key=layout.inference_job_key,
        )
    except (OSError, TypeError, ValueError, FileNotFoundError) as exc:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Could not load release bundle: {exc}",
        )

    ok_compat, compat_msg = check_runtime_bundle_compatibility(effective, spark=spark)
    if not ok_compat:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            message=compat_msg,
        )
    LOGGER.info("Runtime bundle compatibility check passed.")

    try:
        job = load_inference_job_definition(
            inference_yml_path(release_dir, layout.inference_yml_snapshot_rel),
            job_key=layout.inference_job_key,
        )
        resolve_versioned_job_parameters(
            job,
            release_dir,
            launcher_overrides=inputs.param_overrides,
            extra_overrides=inputs.extra_param_overrides,
            stable_trigger=inputs.stable_trigger,
            logger=LOGGER,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            archived_pipeline_version=archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Parameter contract validation failed: {exc}",
        )

    LOGGER.info(
        "Bundle and parameter contract OK at %s (steps=%s, archived_pipeline_version=%s)",
        release_dir,
        effective.get("expected_steps"),
        archived_pipeline_version,
    )
    record_versioned_inference_launcher_event(
        catalog=db_ws,
        event="started",
        databricks_institution_name=inst,
        model_name=model,
        model_run_id=model_run_id,
        archived_pipeline_version=archived_pipeline_version,
        launcher_run_id=launcher_run_id,
        cohort_dataset_name=inputs.param_overrides.get("cohort_file_name"),
        course_dataset_name=inputs.param_overrides.get("course_file_name"),
        payload={"task": "versioned_inference_launcher_validate", "validated": True},
        logger=LOGGER,
    )
    return 0
