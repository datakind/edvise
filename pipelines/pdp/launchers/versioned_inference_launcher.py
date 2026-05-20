#!/usr/bin/env python3
"""
MVP launcher: validate a **versioned runtime bundle** before Git-based inference submit.

Release layout (``<release_base>/<pipeline_version>/``, dev: SHA-named folder):

- ``databricks_bundle_snapshot/resources/github_pdp_inference.yml`` — archived DAB
  inference job; **parsed at run time** for steps, DBR, libraries.

This task only checks that the bundle exists and that the cluster/runtime matches the
archived job definition. Inference runs from Git at ``pipeline_version`` via
``trigger_versioned_inference``; no wheel, ``release.json``, ``pyproject.toml``, or
``release_requirements.txt`` are used.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import inspect
from pathlib import Path
from typing import Any, Mapping


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
    build_effective_release,
)
from pipelines.pdp.launchers.model_metadata import (  # noqa: E402
    escape_sql_string_literal,
    get_spark_session,
    pipeline_version_from_config_toml,
    pipeline_version_from_payload_json_str,
    resolve_model_run_and_pipeline_version,
    resolve_release_dir,
    silver_training_config_path,
    sql_select_latest_pipeline_model,
)

LOGGER = logging.getLogger("versioned_inference_launcher")

DEFAULT_RELEASE_BASE = "/Volumes/dev_sst_02/default/edvise_releases"

BUNDLE_ARCHIVED_DAB_HINT = (
    "This Edvise runtime bundle is not compatible with the current cluster/runtime. "
    "Run inference using the archived Databricks bundle for this release "
    "(see databricks_bundle_snapshot/ in the bundle; MVP does not auto-trigger jobs)."
)


_DBR_MAJOR_MINOR = re.compile(r"^(\d+)\.(\d+)")


def dbr_major_minor(version: str) -> tuple[int, int] | None:
    """Parse DBR ``major.minor`` from ``15.4`` or ``15.4.x-cpu-ml-scala2.12``."""
    m = _DBR_MAJOR_MINOR.match(version.strip().lower())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def databricks_runtime_compatible(required: str, current: str) -> bool:
    """
    Return whether cluster DBR matches the bundle requirement.

    ``DATABRICKS_RUNTIME_VERSION`` is often shortened (e.g. ``15.4``) while job YAML
    uses the full image key (``15.4.x-cpu-ml-scala2.12``).
    """
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
    """Parse a ``major.minor`` Python requirement (e.g. ``3.11``)."""
    s = spec.strip()
    parts = s.replace(" ", "").split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def current_databricks_runtime_version() -> str | None:
    """Databricks cluster image tag, when available."""
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
    """
    Validate driver Python / DBR against ``effective["required_runtime"]`` (from DAB YAML).
    """
    mode = effective.get("execution_mode")
    if isinstance(mode, str) and mode.strip().lower() == "dab":
        return (
            False,
            "This bundle declares execution_mode=dab; use archived DAB job for this "
            "release (wheel launcher path not used). "
            + BUNDLE_ARCHIVED_DAB_HINT,
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
                "is unset; skipping DBR check (local or non-Databricks).",
                req_dbr,
            )
        elif not databricks_runtime_compatible(req_dbr, cur):
            return (
                False,
                f"Bundle requires DBR {req_dbr!r}; current cluster is {cur!r}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
            )
        else:
            logger.info(
                "DBR compatibility OK (bundle=%r, cluster=%r)",
                req_dbr,
                cur,
            )

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
            "Resolve model metadata and validate archived inference YAML from the release "
            "bundle against the current cluster (Git-based inference submit; no pip/wheel)."
        ),
    )
    parser.add_argument(
        "--databricks_institution_name",
        default="",
        help=(
            "Institution slug matching pipeline_models.institution_id (same as PDP "
            "job parameter databricks_institution_name, e.g. miles_cc)."
        ),
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="Registered model name matching pipeline_models.model_name.",
    )
    parser.add_argument(
        "--DB_workspace",
        default="",
        help=(
            "Unity Catalog catalog / workspace id (e.g. dev_sst_02). Used for "
            "``<DB_workspace>.default.pipeline_models`` and silver_volume paths."
        ),
    )
    parser.add_argument(
        "--release_base_path",
        default=DEFAULT_RELEASE_BASE,
        help=f"Base path for versioned releases (default: {DEFAULT_RELEASE_BASE!r}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)

    inst = (args.databricks_institution_name or "").strip()
    model = (args.model_name or "").strip()
    db_ws = (args.DB_workspace or "").strip()

    if not inst or not model or not db_ws:
        LOGGER.error(
            "Require --databricks_institution_name, --model_name, and --DB_workspace."
        )
        return 1

    spark = get_spark_session()
    if spark is None:
        LOGGER.error(
            "SparkSession is required to read pipeline_models (run this task on Databricks)."
        )
        return 1

    resolved = resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace=db_ws,
        databricks_institution_name=inst,
        model_name=model,
    )
    if resolved is None:
        return 1
    _model_run_id, pipeline_version = resolved

    release_dir = resolve_release_dir(args.release_base_path, pipeline_version)
    LOGGER.info("Release bundle directory: %s", release_dir)
    if not release_dir.is_dir():
        LOGGER.error("Release bundle directory not found: %s", release_dir)
        return 1

    try:
        effective = build_effective_release(release_dir, pipeline_version)
    except (OSError, TypeError, ValueError, FileNotFoundError) as exc:
        LOGGER.error("Could not load release bundle: %s", exc)
        return 1

    ok_compat, compat_msg = check_runtime_bundle_compatibility(effective, spark=spark)
    if not ok_compat:
        LOGGER.error("%s", compat_msg)
        return 1
    LOGGER.info("Runtime bundle compatibility check passed.")

    LOGGER.info(
        "Bundle OK at %s (steps=%s, pipeline_version=%s)",
        release_dir,
        effective.get("expected_steps"),
        pipeline_version,
    )
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
