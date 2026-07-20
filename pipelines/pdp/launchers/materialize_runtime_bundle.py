#!/usr/bin/env python3
"""
Databricks task 1: resolve ``pipeline_version`` and materialize the runtime bundle on UC volume.

Fetches archived DAB YAML from GitHub at the resolved ref (Git SHA in dev, release tag in
staging) into ``<release_base>/<pipeline_version>/databricks_bundle_snapshot/``.
"""

from __future__ import annotations

import argparse
import inspect
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

from pipelines.pdp.launchers.bundle_materialize import (  # noqa: E402
    DEFAULT_GITHUB_REPO,
    inference_yml_in_bundle,
    materialize_runtime_bundle_dir,
)
from pipelines.pdp.launchers.launcher_cli import (  # noqa: E402
    add_model_resolution_args,
    optional_model_run_id,
)
from pipelines.pdp.launchers.launcher_run_metadata import (  # noqa: E402
    resolve_launcher_run_id,
    record_versioned_inference_launcher_event,
)
from pipelines.pdp.launchers.model_metadata import (  # noqa: E402
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)
from pipelines.pdp.launchers.pipeline_version_ref import git_ref_kind  # noqa: E402

LOGGER = logging.getLogger("materialize_runtime_bundle")


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

    spark = get_spark_session()
    if spark is None:
        LOGGER.error("SparkSession is required (run on Databricks).")
        return 1

    launcher_run_id = resolve_launcher_run_id(getattr(args, "launcher_run_id", ""))
    record_versioned_inference_launcher_event(
        catalog=db_ws,
        event="started",
        databricks_institution_name=inst,
        model_name=model,
        launcher_run_id=launcher_run_id,
        logger=LOGGER,
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
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="failed",
            databricks_institution_name=inst,
            model_name=model,
            launcher_run_id=launcher_run_id,
            error_message="Could not resolve model_run_id / pipeline_version",
            logger=LOGGER,
        )
        return 1
    model_run_id, pipeline_version = resolved
    LOGGER.info(
        "Materializing bundle for model_run_id=%s pipeline_version=%s (git %s)",
        model_run_id,
        pipeline_version,
        git_ref_kind(pipeline_version),
    )

    from pipelines.pdp.launchers.release_config import resolve_release_base_path

    release_base = resolve_release_base_path(db_ws, args.release_base_path)
    release_dir = resolve_release_dir(release_base, pipeline_version)
    try:
        materialize_runtime_bundle_dir(
            release_dir,
            pipeline_version,
            git_ref=pipeline_version,
            github_repo=args.github_repo.strip() or DEFAULT_GITHUB_REPO,
            skip_snapshot_if_present=args.skip_snapshot_if_present,
            logger=LOGGER,
        )
    except (OSError, ValueError) as exc:
        LOGGER.error("Failed to materialize runtime bundle: %s", exc)
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="failed",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            error_message=str(exc),
            logger=LOGGER,
        )
        return 1

    marker = inference_yml_in_bundle(release_dir)
    if not marker.is_file():
        msg = f"DAB snapshot missing after materialize: {marker}"
        LOGGER.error(msg)
        record_versioned_inference_launcher_event(
            catalog=db_ws,
            event="failed",
            databricks_institution_name=inst,
            model_name=model,
            model_run_id=model_run_id,
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            error_message=msg,
            logger=LOGGER,
        )
        return 1

    LOGGER.info("Runtime bundle materialized at %s", release_dir)
    record_versioned_inference_launcher_event(
        catalog=db_ws,
        event="started",
        databricks_institution_name=inst,
        model_name=model,
        model_run_id=model_run_id,
        pipeline_version=pipeline_version,
        launcher_run_id=launcher_run_id,
        payload={
            "bundle_materialized": str(release_dir),
            "task": "materialize_runtime_bundle",
        },
        logger=LOGGER,
    )
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
