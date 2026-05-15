#!/usr/bin/env python3
"""
Databricks task 1: resolve ``pipeline_version`` and materialize the runtime bundle on UC volume.

Fetches archived DAB YAML from GitHub at the resolved SHA (dev) into
``<release_base>/<pipeline_version>/databricks_bundle_snapshot/`` and writes minimal
``release.json``. Upload the versioned ``*.whl`` to the same folder before or after
this task; the launcher task discovers it via ``release.json`` or a single ``*.whl``.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path


def _setup_import_path() -> None:
    """Databricks may exec this file without ``__file__``; use stack trace to find launchers/."""
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

from pipelines.pdp.launchers.bundle_from_dab import (  # noqa: E402
    DEFAULT_ENTRYPOINT,
    discover_wheel_filename,
)
from pipelines.pdp.launchers.bundle_materialize import (  # noqa: E402
    DEFAULT_GITHUB_REPO,
    inference_yml_in_bundle,
    materialize_runtime_bundle_dir,
)
from pipelines.pdp.launchers.model_metadata import (  # noqa: E402
    get_spark_session,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
)

LOGGER = logging.getLogger("materialize_runtime_bundle")
DEFAULT_RELEASE_BASE = "/Volumes/dev_sst_02/default/edvise_releases"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve pipeline_version and materialize runtime bundle "
            "(DAB snapshot from GitHub + release.json) on the release volume."
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
        "--github_repo",
        default=DEFAULT_GITHUB_REPO,
        help="GitHub org/repo for raw YAML fetch (default: datakind/edvise).",
    )
    parser.add_argument(
        "--wheel",
        default="",
        help="Wheel filename if already uploaded to the release folder (optional).",
    )
    parser.add_argument(
        "--entrypoint",
        default=DEFAULT_ENTRYPOINT,
    )
    parser.add_argument(
        "--skip-snapshot-if-present",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip GitHub fetch when inference YAML snapshot already exists.",
    )
    parser.add_argument(
        "--require-wheel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if no wheel is present in the release folder after materialize.",
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
        "Materializing bundle for model_run_id=%s pipeline_version=%s",
        model_run_id,
        pipeline_version,
    )

    release_dir = resolve_release_dir(args.release_base_path, pipeline_version)
    wheel_hint = (args.wheel or "").strip() or None
    try:
        materialize_runtime_bundle_dir(
            release_dir,
            pipeline_version,
            git_sha=pipeline_version,
            github_repo=args.github_repo.strip() or DEFAULT_GITHUB_REPO,
            wheel=wheel_hint,
            entrypoint=(args.entrypoint or "").strip() or DEFAULT_ENTRYPOINT,
            skip_snapshot_if_present=args.skip_snapshot_if_present,
            logger=LOGGER,
        )
    except (OSError, ValueError) as exc:
        LOGGER.error("Failed to materialize runtime bundle: %s", exc)
        return 1

    if args.require_wheel:
        wheel_name = wheel_hint or discover_wheel_filename(release_dir, None)
        if not wheel_name:
            LOGGER.error(
                "No wheel in %s — upload *.whl before running the launcher task.",
                release_dir,
            )
            return 1

    marker = inference_yml_in_bundle(release_dir)
    if not marker.is_file():
        LOGGER.error("DAB snapshot missing after materialize: %s", marker)
        return 1

    LOGGER.info("Runtime bundle materialized at %s", release_dir)
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
