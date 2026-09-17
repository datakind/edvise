"""ES launcher task 2: validate archived ES (+ optional GenAI) bundles and params."""

from __future__ import annotations

import argparse
import logging
import sys

from edvise.runtime.versioned_inference.bundle.from_dab import (
    build_effective_release,
    inference_yml_path,
    load_inference_job_definition,
)
from edvise.runtime.versioned_inference.cli import (
    add_es_inference_trigger_args,
    build_es_launcher_trigger_inputs,
    optional_model_run_id,
    parse_is_genai_institution,
)
from edvise.runtime.versioned_inference.dab_layout import (
    genai_execute_dab_bundle_layout,
    genai_snapshot_dir,
    resolve_dab_bundle_layout,
)
from edvise.runtime.versioned_inference.genai_registry import (
    resolve_genai_pipeline_version_from_registry,
)
from edvise.runtime.versioned_inference.model_resolution import (
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)
from edvise.runtime.versioned_inference.parameters import (
    resolve_versioned_job_parameters,
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
from edvise.runtime.versioned_inference.submit import DEFAULT_GIT_URL

LOGGER = logging.getLogger("versioned_inference_launcher_validate_es")
TASK_NAME = "versioned_inference_launcher_validate"
_ES_SCHEMA_TYPE = "edvise"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate archived ES inference bundle (and GenAI snapshot when required), "
            "cluster compatibility, and parameter contract."
        ),
    )
    add_es_inference_trigger_args(parser)
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
    with record_launcher_failures(
        catalog=db_ws,
        databricks_institution_name=inst,
        model_name=model,
        launcher_run_id=launcher_run_id,
        task=TASK_NAME,
        logger=LOGGER,
    ) as event:
        # Force edvise schema for layout/overrides even if job param is blank.
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
        es_layout = resolve_dab_bundle_layout(_ES_SCHEMA_TYPE)

        release_dir = resolve_release_dir(inputs.release_base_path, es_pipeline_version)
        LOGGER.info(
            "ES release bundle directory: %s (es_pipeline_version=%s, git %s)",
            release_dir,
            es_pipeline_version,
            git_ref_kind(es_pipeline_version),
        )
        if not release_dir.is_dir():
            raise FileNotFoundError(
                f"ES release bundle directory not found: {release_dir}. "
                "Run materialize_runtime_bundle first."
            )

        es_effective = build_effective_release(
            release_dir,
            es_pipeline_version,
            inference_yml_relative=es_layout.inference_yml_snapshot_rel,
            inference_job_key=es_layout.inference_job_key,
        )
        ok_compat, compat_msg = check_runtime_bundle_compatibility(
            es_effective, spark=spark
        )
        if not ok_compat:
            raise RuntimeError(compat_msg)
        LOGGER.info("ES runtime bundle compatibility check passed.")

        es_job = load_inference_job_definition(
            inference_yml_path(release_dir, es_layout.inference_yml_snapshot_rel),
            job_key=es_layout.inference_job_key,
        )
        # Archived ES contract references pipeline_version for the nested GenAI call;
        # for the dual-pin design the GenAI commit comes from the registry, but the ES
        # contract still needs a concrete value during validate.
        param_overrides = dict(inputs.param_overrides)
        param_overrides["pipeline_version"] = es_pipeline_version
        resolve_versioned_job_parameters(
            es_job,
            release_dir,
            launcher_overrides=param_overrides,
            extra_overrides=inputs.extra_param_overrides,
            stable_trigger=inputs.stable_trigger,
            logger=LOGGER,
        )

        genai_pipeline_version: str | None = None
        genai_dir = None
        if is_genai:
            genai_pipeline_version = resolve_genai_pipeline_version_from_registry(
                db_ws,
                inst,
                logger=LOGGER,
            )
            genai_layout = genai_execute_dab_bundle_layout()
            genai_dir = genai_snapshot_dir(release_dir)
            if not genai_dir.is_dir():
                raise FileNotFoundError(
                    f"GenAI release bundle directory not found: {genai_dir}. "
                    "Run materialize_runtime_bundle with is_genai_institution=true."
                )
            genai_marker = inference_yml_path(
                genai_dir, genai_layout.inference_yml_snapshot_rel
            )
            if not genai_marker.is_file():
                raise FileNotFoundError(f"GenAI DAB snapshot missing: {genai_marker}")

            genai_effective = build_effective_release(
                genai_dir,
                genai_pipeline_version,
                inference_yml_relative=genai_layout.inference_yml_snapshot_rel,
                inference_job_key=genai_layout.inference_job_key,
            )
            ok_genai, genai_compat_msg = check_runtime_bundle_compatibility(
                genai_effective, spark=spark
            )
            if not ok_genai:
                raise RuntimeError(genai_compat_msg)
            LOGGER.info(
                "GenAI runtime bundle compatibility check passed "
                "(genai_pipeline_version=%s, git %s).",
                genai_pipeline_version,
                git_ref_kind(genai_pipeline_version),
            )

            genai_job = load_inference_job_definition(
                genai_marker,
                job_key=genai_layout.inference_job_key,
            )
            # GenAI execute contract uses nested names; supply mapped ES values so
            # required refs resolve during validate (trigger will re-resolve).
            genai_overrides = {
                "institution_id": inst,
                "catalog": db_ws,
                "pipeline_version": genai_pipeline_version,
                "db_run_id": launcher_run_id or param_overrides.get("db_run_id", ""),
                "inputs_toml_path": param_overrides.get(
                    "genai_inputs_toml_path", "inputs.toml"
                ),
                # bronze_batch_dir is produced by segment-1 ingestion; allow empty here.
                "bronze_batch_dir": "",
            }
            resolve_versioned_job_parameters(
                genai_job,
                genai_dir,
                launcher_overrides=genai_overrides,
                extra_overrides=None,
                stable_trigger=None,
                logger=LOGGER,
            )

        payload: dict[str, object] = {
            "task": TASK_NAME,
            "validated": True,
            "schema_type": _ES_SCHEMA_TYPE,
            "is_genai_institution": is_genai,
            "es_steps": es_effective.get("expected_steps"),
            "bundle_materialized": str(release_dir),
        }
        if genai_pipeline_version is not None:
            payload["genai_pipeline_version"] = genai_pipeline_version
        if genai_dir is not None:
            payload["genai_bundle_materialized"] = str(genai_dir)

        LOGGER.info(
            "ES bundle and parameter contract OK at %s "
            "(es_steps=%s, es_pipeline_version=%s, is_genai=%s)",
            release_dir,
            es_effective.get("expected_steps"),
            es_pipeline_version,
            is_genai,
        )
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="started",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            archived_pipeline_version=es_pipeline_version,
            launcher_run_id=launcher_run_id,
            payload=payload,
            logger=LOGGER,
        )
