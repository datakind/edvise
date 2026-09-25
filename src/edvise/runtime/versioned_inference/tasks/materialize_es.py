"""ES launcher task 1: materialize ES (+ optional GenAI) DAB snapshots on UC volume."""

from __future__ import annotations

import argparse
import logging
import sys

from edvise.dataio.batch_gcs_inference_ingest import parse_is_genai_institution
from edvise.runtime.versioned_inference.bundle.from_dab import inference_yml_path
from edvise.runtime.versioned_inference.bundle.materialize import (
    DEFAULT_GITHUB_REPO,
    materialize_runtime_bundle_dir,
)
from edvise.runtime.versioned_inference.cli import (
    add_model_resolution_args,
    optional_model_run_id,
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
from edvise.runtime.versioned_inference.pipeline_version_ref import git_ref_kind
from edvise.runtime.versioned_inference.release_config import (
    resolve_es_release_base_path,
)
from edvise.runtime.versioned_inference.run_metadata import (
    record_launcher_failures,
    record_versioned_inference_launcher_event,
    resolve_launcher_run_id,
)

LOGGER = logging.getLogger("materialize_es_runtime_bundle")
TASK_NAME = "materialize_runtime_bundle"
_ES_SCHEMA_TYPE = "edvise"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve ES pipeline_version and materialize ES (+ optional GenAI) "
            "DAB YAML snapshots under edvise_releases/es/{version}/."
        ),
    )
    add_model_resolution_args(parser)
    parser.add_argument(
        "--github_repo",
        default=DEFAULT_GITHUB_REPO,
        help="GitHub org/repo for raw YAML fetch (default: datakind/edvise).",
    )
    parser.add_argument(
        "--is_genai_institution",
        default="false",
        help='When "true", also materialize GenAI execute YAML under genai/.',
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

    is_genai = parse_is_genai_institution(getattr(args, "is_genai_institution", ""))
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
        model_run_id, es_pipeline_version = resolved
        event.model_run_id = model_run_id
        event.archived_pipeline_version = es_pipeline_version
        LOGGER.info(
            "Materializing ES bundle for model_run_id=%s es_pipeline_version=%s (git %s)",
            model_run_id,
            es_pipeline_version,
            git_ref_kind(es_pipeline_version),
        )

        release_base = resolve_es_release_base_path(db_ws, args.release_base_path)
        release_dir = resolve_release_dir(release_base, es_pipeline_version)
        es_layout = resolve_dab_bundle_layout(_ES_SCHEMA_TYPE)
        github_repo = args.github_repo.strip() or DEFAULT_GITHUB_REPO
        materialize_runtime_bundle_dir(
            release_dir,
            es_pipeline_version,
            schema_type=_ES_SCHEMA_TYPE,
            git_ref=es_pipeline_version,
            github_repo=github_repo,
            skip_snapshot_if_present=args.skip_snapshot_if_present,
            logger=LOGGER,
        )

        es_marker = inference_yml_path(
            release_dir, es_layout.inference_yml_snapshot_rel
        )
        if not es_marker.is_file():
            raise FileNotFoundError(
                f"ES DAB snapshot missing after materialize: {es_marker}"
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
            LOGGER.info(
                "Materializing GenAI execute bundle at %s (git %s) under %s",
                genai_pipeline_version,
                git_ref_kind(genai_pipeline_version),
                genai_dir,
            )
            materialize_runtime_bundle_dir(
                genai_dir,
                genai_pipeline_version,
                layout=genai_layout,
                git_ref=genai_pipeline_version,
                github_repo=github_repo,
                skip_snapshot_if_present=args.skip_snapshot_if_present,
                logger=LOGGER,
            )
            genai_marker = inference_yml_path(
                genai_dir, genai_layout.inference_yml_snapshot_rel
            )
            if not genai_marker.is_file():
                raise FileNotFoundError(
                    f"GenAI DAB snapshot missing after materialize: {genai_marker}"
                )

        payload: dict[str, object] = {
            "bundle_materialized": str(release_dir),
            "task": TASK_NAME,
            "schema_type": es_layout.schema_type,
            "is_genai_institution": is_genai,
        }
        if genai_pipeline_version is not None:
            payload["genai_pipeline_version"] = genai_pipeline_version
        if genai_dir is not None:
            payload["genai_bundle_materialized"] = str(genai_dir)

        LOGGER.info("ES runtime bundle materialized at %s", release_dir)
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="completed",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            archived_pipeline_version=es_pipeline_version,
            launcher_run_id=launcher_run_id,
            payload=payload,
            logger=LOGGER,
        )
