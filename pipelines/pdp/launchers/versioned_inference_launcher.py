#!/usr/bin/env python3
"""
Validate a versioned runtime bundle and resolve inference parameters before child submit.

Checks DBR/Python compatibility and runs full parameter contract resolution using the
same inputs as ``trigger_versioned_inference``. Fails closed — no fallback to HEAD inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import inspect
from pathlib import Path
from typing import Any, Mapping


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

from pipelines.pdp.launchers.bundle_from_dab import (  # noqa: E402
    build_effective_release,
    inference_yml_path,
    load_inference_job_definition,
)
from pipelines.pdp.launchers.inference_parameters import (  # noqa: E402
    resolve_versioned_job_parameters,
)
from pipelines.pdp.launchers.launcher_cli import (  # noqa: E402
    add_inference_trigger_args,
    build_launcher_trigger_inputs,
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
from pipelines.pdp.launchers.inference_job_submit import DEFAULT_GIT_URL  # noqa: E402
from pipelines.pdp.launchers.pipeline_version_ref import git_ref_kind  # noqa: E402

LOGGER = logging.getLogger("versioned_inference_launcher")

BUNDLE_ARCHIVED_DAB_HINT = (
    "Versioned inference cannot run on this cluster/runtime. "
    "Use the archived Databricks bundle for this pipeline_version."
)

_DBR_MAJOR_MINOR = re.compile(r"^(\d+)\.(\d+)")


def dbr_major_minor(version: str) -> tuple[int, int] | None:
    m = _DBR_MAJOR_MINOR.match(version.strip().lower())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def databricks_runtime_compatible(required: str, current: str) -> bool:
    req = required.strip().lower()
    cur = current.strip().lower()
    if req == cur:
        return True
    req_mm = dbr_major_minor(req)
    cur_mm = dbr_major_minor(cur)
    if req_mm and cur_mm and req_mm == cur_mm:
        return True
    return req.startswith(f"{cur}.") or req.startswith(f"{cur}-")


def parse_python_xy(spec: str) -> tuple[int, int] | None:
    s = spec.strip()
    parts = s.replace(" ", "").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def current_databricks_runtime_version() -> str | None:
    v = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    return v.strip() if isinstance(v, str) and v.strip() else None


def current_spark_version(spark: Any) -> str | None:
    try:
        v = spark.version
        return str(v).strip() if v else None
    except Exception:
        return None


def check_runtime_bundle_compatibility(
    effective: Mapping[str, Any],
    *,
    spark: Any,
    logger: logging.Logger = LOGGER,
) -> tuple[bool, str]:
    mode = effective.get("execution_mode")
    if isinstance(mode, str) and mode.strip().lower() == "dab":
        return (
            False,
            "This bundle declares execution_mode=dab; use archived DAB job for this "
            "release. " + BUNDLE_ARCHIVED_DAB_HINT,
        )

    rr = effective.get("required_runtime")
    if not isinstance(rr, dict):
        return True, ""

    req_py = rr.get("python")
    if isinstance(req_py, str) and req_py.strip():
        want = parse_python_xy(req_py)
        got = sys.version_info[:2]
        if want and got != want:
            return (
                False,
                f"Bundle requires Python {req_py}; driver is {got[0]}.{got[1]}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )

    req_dbr = rr.get("databricks_runtime")
    if isinstance(req_dbr, str) and req_dbr.strip():
        cur = current_databricks_runtime_version()
        if not cur:
            logger.warning(
                "Bundle requires databricks_runtime=%r but DATABRICKS_RUNTIME_VERSION "
                "is unset; skipping DBR check.",
                req_dbr,
            )
        elif not databricks_runtime_compatible(req_dbr, cur):
            return (
                False,
                f"Bundle requires DBR {req_dbr!r}; current cluster is {cur!r}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )
        logger.info("DBR compatibility OK (bundle=%r, cluster=%r)", req_dbr, cur)

    req_spark = rr.get("spark")
    if isinstance(req_spark, str) and req_spark.strip():
        cur_sp = current_spark_version(spark)
        if cur_sp and cur_sp.strip() != req_spark.strip():
            return (
                False,
                f"Bundle requires Spark {req_spark!r}; active Spark is {cur_sp!r}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )

    return True, ""


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

    release_dir = resolve_release_dir(inputs.release_base_path, pipeline_version)
    LOGGER.info(
        "Release bundle directory: %s (pipeline_version=%s, git %s)",
        release_dir,
        pipeline_version,
        git_ref_kind(pipeline_version),
    )
    if not release_dir.is_dir():
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Release bundle directory not found: {release_dir}",
        )

    try:
        effective = build_effective_release(release_dir, pipeline_version)
    except (OSError, TypeError, ValueError, FileNotFoundError) as exc:
        return _fail(
            catalog=db_ws,
            inst=inst,
            model=model,
            model_run_id=model_run_id,
            pipeline_version=pipeline_version,
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
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            message=compat_msg,
        )
    LOGGER.info("Runtime bundle compatibility check passed.")

    try:
        job = load_inference_job_definition(inference_yml_path(release_dir))
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
            pipeline_version=pipeline_version,
            launcher_run_id=launcher_run_id,
            message=f"Parameter contract validation failed: {exc}",
        )

    LOGGER.info(
        "Bundle and parameter contract OK at %s (steps=%s, pipeline_version=%s)",
        release_dir,
        effective.get("expected_steps"),
        pipeline_version,
    )
    record_versioned_inference_launcher_event(
        catalog=db_ws,
        event="started",
        databricks_institution_name=inst,
        model_name=model,
        model_run_id=model_run_id,
        pipeline_version=pipeline_version,
        launcher_run_id=launcher_run_id,
        cohort_dataset_name=inputs.param_overrides.get("cohort_file_name"),
        course_dataset_name=inputs.param_overrides.get("course_file_name"),
        payload={"task": "versioned_inference_launcher_validate", "validated": True},
        logger=LOGGER,
    )
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
