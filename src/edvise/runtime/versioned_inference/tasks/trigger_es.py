"""ES launcher task 3: dual-pin / ES-full versioned inference submit."""

from __future__ import annotations

import argparse
import logging
import sys

from edvise.runtime.versioned_inference.cli import (
    add_es_inference_trigger_args,
    build_es_launcher_trigger_inputs,
    optional_model_run_id,
    parse_is_genai_institution,
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
from edvise.runtime.versioned_inference.submit import DEFAULT_GIT_URL
from edvise.runtime.versioned_inference.submit_es import (
    submit_es_versioned_inference_from_bundle,
)

LOGGER = logging.getLogger("trigger_versioned_inference_es")
TASK_NAME = "trigger_versioned_inference"
_ES_SCHEMA_TYPE = "edvise"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit versioned ES inference from archived bundle YAML. "
            "Non-GenAI schools: one ES-full spark child run at the model ES "
            "pipeline_version. GenAI schools: three dual-pin child runs "
            "(ES prefix → GenAI @ registry → ES suffix)."
        ),
    )
    add_es_inference_trigger_args(parser)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Do not wait for the final child run. GenAI path still waits for "
            "prefix and GenAI execute so handoff can proceed."
        ),
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
        help="Log submit payload(s) without calling the Jobs API.",
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

    is_genai = parse_is_genai_institution(getattr(args, "is_genai_institution", ""))
    launcher_run_id = resolve_launcher_run_id(getattr(args, "launcher_run_id", ""))
    if not launcher_run_id:
        raise ValueError(
            "launcher_run_id is required (job parameter launcher_run_id with "
            "default {{job.run_id}})."
        )

    with record_launcher_failures(
        catalog=db_ws,
        databricks_institution_name=inst,
        model_name=model,
        launcher_run_id=launcher_run_id,
        task=TASK_NAME,
        logger=LOGGER,
    ) as event:
        args.schema_type = _ES_SCHEMA_TYPE
        inputs = build_es_launcher_trigger_inputs(args, default_git_url=DEFAULT_GIT_URL)

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
        model_run_id, es_pipeline_version = resolved
        event.model_run_id = model_run_id
        event.archived_pipeline_version = es_pipeline_version

        LOGGER.info(
            "Triggering ES inference for model_run_id=%s es_pipeline_version=%s "
            "(git %s, is_genai=%s)",
            model_run_id,
            es_pipeline_version,
            git_ref_kind(es_pipeline_version),
            is_genai,
        )

        release_dir = resolve_release_dir(inputs.release_base_path, es_pipeline_version)
        if not release_dir.is_dir():
            raise FileNotFoundError(
                f"Release bundle not found: {release_dir} "
                "(run materialize_runtime_bundle first)"
            )

        child_ids = submit_es_versioned_inference_from_bundle(
            release_dir,
            es_pipeline_version=es_pipeline_version,
            is_genai=is_genai,
            parameter_overrides=inputs.param_overrides,
            extra_parameter_overrides=inputs.extra_param_overrides,
            stable_trigger=inputs.stable_trigger,
            git_url=inputs.git_url,
            db_workspace=db_ws,
            databricks_institution_name=inst,
            model_run_id=model_run_id,
            dry_run=args.dry_run,
            wait_for_completion=not args.no_wait,
            poll_interval_seconds=args.poll_interval_seconds,
            logger=LOGGER,
        )

        if args.dry_run:
            return

        db_run_id = inputs.param_overrides.get("db_run_id") or launcher_run_id
        if inputs.extra_param_overrides.get("db_run_id"):
            db_run_id = inputs.extra_param_overrides["db_run_id"]

        payload: dict[str, object] = {
            "task": TASK_NAME,
            "no_wait": args.no_wait,
            "schema_type": _ES_SCHEMA_TYPE,
            "is_genai_institution": is_genai,
            "parent_launcher_run_id": launcher_run_id,
            "db_run_id": db_run_id,
            **child_ids.as_payload(),
        }

        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="completed" if not args.no_wait else "started",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            archived_pipeline_version=es_pipeline_version,
            launcher_run_id=launcher_run_id,
            child_inference_run_id=child_ids.primary,
            payload=payload,
            logger=LOGGER,
        )
        # Final summary: ids as structured payload, then URLs alone (clickable).
        LOGGER.info(
            "ES versioned inference %s (training model_run_id=%s, "
            "parent_launcher_run_id=%s, db_run_id=%s, child_ids=%s)",
            "submitted (no-wait)" if args.no_wait else "completed successfully",
            model_run_id,
            launcher_run_id,
            db_run_id,
            child_ids.as_payload(),
        )
        child_ids.log_monitor_urls(LOGGER)
