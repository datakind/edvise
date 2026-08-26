"""Task 2: validate archived bundle and parameter contract."""

from __future__ import annotations

import argparse
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
    record_launcher_failures,
    record_versioned_inference_launcher_event,
    resolve_launcher_run_id,
)
from edvise.runtime.versioned_inference.runtime_compat import (
    check_runtime_bundle_compatibility,
)

LOGGER = logging.getLogger("versioned_inference_launcher")
TASK_NAME = "versioned_inference_launcher_validate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate archived inference bundle, cluster compatibility, and parameter contract."
        ),
    )
    add_inference_trigger_args(parser)
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
    with record_launcher_failures(
        catalog=db_ws,
        databricks_institution_name=inst,
        model_name=model,
        launcher_run_id=launcher_run_id,
        task=TASK_NAME,
        logger=LOGGER,
    ) as event:
        inputs = build_launcher_trigger_inputs(args, default_git_url=DEFAULT_GIT_URL)

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
        layout = resolve_dab_bundle_layout(inputs.schema_type)

        release_dir = resolve_release_dir(
            inputs.release_base_path, archived_pipeline_version
        )
        LOGGER.info(
            "Release bundle directory: %s (archived_pipeline_version=%s, git %s)",
            release_dir,
            archived_pipeline_version,
            git_ref_kind(archived_pipeline_version),
        )
        if not release_dir.is_dir():
            raise FileNotFoundError(
                f"Release bundle directory not found: {release_dir}"
            )

        effective = build_effective_release(
            release_dir,
            archived_pipeline_version,
            inference_yml_relative=layout.inference_yml_snapshot_rel,
            inference_job_key=layout.inference_job_key,
        )

        ok_compat, compat_msg = check_runtime_bundle_compatibility(
            effective, spark=spark
        )
        if not ok_compat:
            raise RuntimeError(compat_msg)
        LOGGER.info("Runtime bundle compatibility check passed.")

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
            payload={"task": TASK_NAME, "validated": True},
            logger=LOGGER,
        )
