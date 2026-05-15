#!/usr/bin/env python3
"""
MVP launcher: stable dispatcher for a **versioned runtime bundle**.

Release layout (``<release_base>/<pipeline_version>/``, dev: SHA-named folder):

- ``release.json`` — optional overrides only (``wheel``, ``entrypoint``, …).
- ``*.whl`` — Edvise wheel for that release.
- ``databricks_bundle_snapshot/resources/github_pdp_inference.yml`` — archived DAB
  inference job; **parsed at run time** for steps, DBR, libraries, job parameters.

The launcher does not import ``edvise`` before installing the bundle wheel.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
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
    DEFAULT_ENTRYPOINT,
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
        elif cur.strip().lower() != req_dbr.strip().lower():
            return (
                False,
                f"Bundle requires DBR {req_dbr!r}; current cluster is {cur!r}. "
                + BUNDLE_ARCHIVED_DAB_HINT,
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


def validate_required_payload_fields(
    effective: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[bool, str]:
    """Ensure payload contains every key listed in ``required_payload_fields``."""
    fields = effective.get("required_payload_fields")
    if fields is None:
        return True, ""
    if not isinstance(fields, list):
        return True, ""
    missing = [
        f
        for f in fields
        if isinstance(f, str) and f.strip() and f.strip() not in payload
    ]
    if missing:
        return (
            False,
            f"Payload missing required fields from release bundle: {missing!r}",
        )
    return True, ""


def build_payload_dict(
    model_run_id: str,
    pipeline_version: str,
    release: Mapping[str, Any],
    payload_json: dict[str, Any] | None,
    *,
    databricks_institution_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build the payload object written for the versioned entrypoint."""
    out: dict[str, Any] = dict(payload_json or {})
    out["model_run_id"] = model_run_id
    out["pipeline_version"] = pipeline_version
    out["release"] = dict(release)
    if databricks_institution_name:
        out["databricks_institution_name"] = databricks_institution_name
    if model_name:
        out["model_name"] = model_name
    return out


def resolve_wheel_path(release_dir: Path, wheel_name: str) -> Path:
    """Absolute path to the wheel file under the release directory."""
    return release_dir / wheel_name


def pip_install_wheel_command(python_executable: str, wheel_path: str | Path) -> list[str]:
    """Command to force-reinstall the wheel (MVP; no dependency locking)."""
    return [
        python_executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        str(wheel_path),
    ]


def verify_edvise_import_command(python_executable: str) -> list[str]:
    """Command that prints ``edvise.__file__`` after install (proves site-packages)."""
    return [
        python_executable,
        "-c",
        "import edvise; print(edvise.__file__)",
    ]


def entrypoint_command(
    python_executable: str, entrypoint: str, payload_path: str | Path
) -> list[str]:
    """Command to run ``python -m <entrypoint> --payload <path>``."""
    return [
        python_executable,
        "-m",
        entrypoint,
        "--payload",
        str(payload_path),
    ]


def default_payload_dir() -> Path:
    """Writable directory for payload files (Databricks: ``/tmp``)."""
    tmp = Path("/tmp")
    if tmp.is_dir():
        return tmp
    return Path(tempfile.gettempdir())


def write_payload_file(payload: Mapping[str, Any], base_dir: Path | None = None) -> Path:
    """Write payload JSON and return its path."""
    directory = base_dir if base_dir is not None else default_payload_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"edvise_versioned_inference_payload_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_logged_subprocess(
    cmd: list[str],
    *,
    label: str,
    logger: logging.Logger = LOGGER,
) -> int:
    """Run a subprocess; log stdout/stderr; return exit code."""
    logger.info("Running %s: %s", label, " ".join(cmd))
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        logger.info("[%s stdout]\n%s", label, completed.stdout.rstrip())
    if completed.stderr:
        logger.info("[%s stderr]\n%s", label, completed.stderr.rstrip())
    if completed.returncode != 0:
        logger.error(
            "%s failed with exit code %s",
            label,
            completed.returncode,
        )
    else:
        logger.info("%s completed successfully", label)
    return completed.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MVP stable launcher: resolve model metadata, load runtime bundle "
            "(minimal release.json + parsed databricks_bundle_snapshot YAML), "
            "install wheel, run versioned entrypoint."
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
    parser.add_argument(
        "--payload_json",
        default=None,
        help="Optional JSON object merged into the payload (merged before core keys).",
    )
    parser.add_argument(
        "--entrypoint",
        default=None,
        help="Override release entrypoint (python -m <value>).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for pip, verify, and entrypoint (default: current interpreter).",
    )
    return parser.parse_args(argv)


def _parse_payload_json(raw: str | None) -> dict[str, Any] | None:
    if raw is None or raw.strip() == "":
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        msg = "--payload_json must decode to a JSON object"
        raise ValueError(msg)
    return data


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    try:
        extra = _parse_payload_json(args.payload_json)
    except (json.JSONDecodeError, ValueError) as exc:
        LOGGER.error("Invalid --payload_json: %s", exc)
        return 1

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
    model_run_id, pipeline_version = resolved

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

    cli_entrypoint = (args.entrypoint or "").strip()
    entrypoint = cli_entrypoint or str(effective.get("entrypoint") or DEFAULT_ENTRYPOINT)
    if not entrypoint:
        LOGGER.error("Entrypoint is empty after release resolution.")
        return 1

    wheel_path = resolve_wheel_path(release_dir, str(effective["wheel"]))
    if not wheel_path.is_file():
        LOGGER.error("Wheel not found: %s", wheel_path)
        return 1

    pip_cmd = pip_install_wheel_command(args.python, wheel_path)
    if run_logged_subprocess(pip_cmd, label="pip_install_wheel") != 0:
        return 1

    verify_cmd = verify_edvise_import_command(args.python)
    if run_logged_subprocess(verify_cmd, label="verify_edvise_import") != 0:
        return 1

    payload = build_payload_dict(
        model_run_id,
        pipeline_version,
        effective,
        extra,
        databricks_institution_name=inst or None,
        model_name=model or None,
    )
    ok_fields, fields_msg = validate_required_payload_fields(effective, payload)
    if not ok_fields:
        LOGGER.error("%s", fields_msg)
        return 1
    try:
        payload_path = write_payload_file(payload)
    except OSError as exc:
        LOGGER.error("Could not write payload file: %s", exc)
        return 1
    LOGGER.info("Wrote payload to %s", payload_path)

    run_cmd = entrypoint_command(args.python, entrypoint, payload_path)
    if run_logged_subprocess(run_cmd, label="versioned_entrypoint") != 0:
        return 1
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
