#!/usr/bin/env python3
"""
MVP launcher: stable dispatcher for a **versioned runtime bundle** (not just a wheel).

A bundle directory is ``<release_base>/<pipeline_version>/`` and should include at least
``manifest.json``, the wheel, optional ``inference_contract.json``, and optional
``databricks_bundle_snapshot/`` (archived DAB slice for audit / future fallback).

In PDP **dev** workspaces, ``pipeline_version`` resolved from ``config.toml`` /
``payload_json`` is typically the **git commit SHA**; the release folder on the volume
uses that same string as the directory name.

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
from pathlib import Path
from typing import Any, Mapping, Protocol

LOGGER = logging.getLogger("versioned_inference_launcher")

DEFAULT_RELEASE_BASE = "/Volumes/dev_sst_02/default/edvise_releases"

BUNDLE_ARCHIVED_DAB_HINT = (
    "This Edvise runtime bundle is not compatible with the current cluster/runtime. "
    "Run inference using the archived Databricks bundle for this release "
    "(see databricks_bundle_snapshot/ in the bundle; MVP does not auto-trigger jobs)."
)


class _SparkSQL(Protocol):
    def sql(self, query: str) -> Any: ...


def escape_sql_string_literal(value: str) -> str:
    """Escape a value for use inside single-quoted SQL string literals."""
    return value.replace("'", "''")


def sql_select_latest_pipeline_model(
    db_workspace: str, institution_id: str, model_name: str
) -> str:
    """
    SQL to load the newest registered model row for an institution and model name.

    ``db_workspace`` is the Unity Catalog catalog (same as PDP ``DB_workspace``, e.g.
    ``dev_sst_02``). ``institution_id`` matches ``pipeline_models.institution_id``
    (same values as ``databricks_institution_name``, e.g. ``miles_cc``).
    """
    cat = db_workspace.replace("`", "")
    inst = escape_sql_string_literal(institution_id)
    model = escape_sql_string_literal(model_name)
    return (
        "SELECT model_run_id, payload_json "
        f"FROM `{cat}`.default.pipeline_models "
        f"WHERE institution_id = '{inst}' AND model_name = '{model}' "
        "ORDER BY logged_ts DESC LIMIT 1"
    )


def silver_training_config_path(
    db_workspace: str, databricks_institution_name: str, model_run_id: str
) -> Path:
    """
    Path to training ``config.toml`` under the institution silver volume for a run.

    Matches PDP layout: ``/Volumes/<ws>/<inst>_silver/silver_volume/<model_run_id>/training/config.toml``.
    """
    return Path(
        f"/Volumes/{db_workspace}/{databricks_institution_name}_silver/"
        f"silver_volume/{model_run_id}/training/config.toml"
    )


def pipeline_version_from_payload_json_str(raw: str | None) -> str | None:
    """Read ``pipeline_version`` from ``pipeline_models.payload_json`` (dev: git commit SHA)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.lower() == "null":
        return None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    v = data.get("pipeline_version")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def pipeline_version_from_config_toml(text: str) -> str | None:
    """Parse ``pipeline_version`` from PDP ``config.toml`` (dev: full git commit SHA)."""
    if sys.version_info >= (3, 11):
        try:
            import tomllib

            data = tomllib.loads(text)
        except Exception:
            data = None
        if isinstance(data, dict):
            v = data.get("pipeline_version")
            if isinstance(v, str) and v.strip():
                return v.strip()
    try:
        import tomli  # type: ignore[import-not-found]

        data = tomli.loads(text)
    except Exception:
        return None
    if isinstance(data, dict):
        v = data.get("pipeline_version")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def get_spark_session() -> Any:
    """Active Spark session on Databricks (None if PySpark is unavailable)."""
    try:
        from pyspark.sql import SparkSession  # type: ignore[import-not-found]

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as exc:
        LOGGER.warning("Could not obtain SparkSession: %s", exc)
        return None


def resolve_model_run_and_pipeline_version(
    *,
    spark: _SparkSQL,
    db_workspace: str,
    databricks_institution_name: str,
    model_name: str,
    logger: logging.Logger = LOGGER,
) -> tuple[str, str] | None:
    """
    Resolve ``model_run_id`` from ``pipeline_models``, then ``pipeline_version``:

    1. Silver ``training/config.toml`` for that run (PDP dev: ``pipeline_version`` is the
       **git commit SHA** used to key the runtime bundle directory).
    2. Else ``payload_json.pipeline_version`` on the model row (same SHA convention when
       config is unavailable).
    """
    q = sql_select_latest_pipeline_model(
        db_workspace, databricks_institution_name, model_name
    )
    logger.info("pipeline_models lookup:\n%s", q)
    rows = spark.sql(q).collect()
    if not rows:
        logger.error(
            "No pipeline_models row for institution_id=%r model_name=%r in %s.default",
            databricks_institution_name,
            model_name,
            db_workspace,
        )
        return None
    row = rows[0]
    model_run_id = str(row["model_run_id"]).strip()
    if not model_run_id:
        logger.error("pipeline_models row has empty model_run_id")
        return None

    cfg_path = silver_training_config_path(
        db_workspace, databricks_institution_name, model_run_id
    )
    pv: str | None = None
    if cfg_path.is_file():
        try:
            pv = pipeline_version_from_config_toml(
                cfg_path.read_text(encoding="utf-8")
            )
            if pv:
                logger.info(
                    "pipeline_version from silver training config.toml (%s): %s",
                    cfg_path,
                    pv,
                )
        except OSError as exc:
            logger.warning("Failed to read config.toml: %s", exc)
    else:
        logger.info(
            "config.toml not found at %s; will try payload_json.pipeline_version",
            cfg_path,
        )

    if not pv:
        payload_raw = row["payload_json"]
        pv = pipeline_version_from_payload_json_str(payload_raw)
        if pv:
            logger.info(
                "pipeline_version from pipeline_models.payload_json (training snapshot): %s",
                pv,
            )

    if not pv:
        logger.error(
            "Could not resolve pipeline_version from %s or payload_json",
            cfg_path,
        )
        return None
    logger.info("Resolved model_run_id=%s pipeline_version=%s", model_run_id, pv)
    return model_run_id, pv


def resolve_manifest_path(release_base_path: str, pipeline_version: str) -> Path:
    """Return ``<release_base_path>/<pipeline_version>/manifest.json`` (dev: SHA-named folder)."""
    return Path(release_base_path).expanduser().resolve() / pipeline_version / "manifest.json"


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and validate release manifest JSON."""
    if not manifest_path.is_file():
        msg = f"Manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "Manifest root must be a JSON object"
        raise TypeError(msg)
    for key in ("pipeline_version", "wheel", "entrypoint"):
        if key not in data:
            msg = f"Manifest missing required key {key!r}"
            raise ValueError(msg)
    return data


def merge_manifest_with_optional_contract(
    manifest: dict[str, Any], release_dir: Path
) -> dict[str, Any]:
    """
    Build effective runtime metadata: ``manifest.json`` overlays optional
    ``inference_contract.json`` when ``manifest["contract"]`` names a sibling file.

    ``required_runtime`` sub-keys from the manifest override the contract file.
    """
    ref = manifest.get("contract")
    if not isinstance(ref, str) or not ref.strip():
        return dict(manifest)
    cpath = (release_dir / ref.strip()).resolve()
    if not cpath.is_file():
        LOGGER.warning(
            "Manifest references contract file %r but it was not found under %s",
            ref,
            release_dir,
        )
        return dict(manifest)
    try:
        raw = json.loads(cpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        LOGGER.warning("Could not read inference contract %s: %s", cpath, exc)
        return dict(manifest)
    if not isinstance(raw, dict):
        return dict(manifest)
    merged: dict[str, Any] = dict(raw)
    merged.update(manifest)
    rr_c = raw.get("required_runtime")
    rr_m = manifest.get("required_runtime")
    if isinstance(rr_c, dict) or isinstance(rr_m, dict):
        rr_out: dict[str, Any] = {}
        if isinstance(rr_c, dict):
            rr_out.update(rr_c)
        if isinstance(rr_m, dict):
            rr_out.update(rr_m)
        merged["required_runtime"] = rr_out
    return merged


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
    Validate driver Python / DBR / Spark against ``effective["required_runtime"]``.

    Missing optional contract fields mean no check for that dimension.
    """
    mode = effective.get("execution_mode")
    if isinstance(mode, str) and mode.strip().lower() == "dab":
        return (
            False,
            "This bundle declares execution_mode=dab; use archived DAB job for this "
            "release (wheel launcher path not used).",
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
            f"Payload missing required fields from runtime bundle contract: {missing!r}",
        )
    return True, ""


def log_bundle_snapshot_presence(release_dir: Path, logger: logging.Logger = LOGGER) -> None:
    snap = release_dir / "databricks_bundle_snapshot"
    if snap.is_dir():
        logger.info(
            "Runtime bundle includes databricks_bundle_snapshot at %s (archived DAB slice).",
            snap,
        )
    else:
        logger.info(
            "No databricks_bundle_snapshot/ under %s (optional for MVP audit trail).",
            release_dir,
        )


def build_payload_dict(
    model_run_id: str,
    pipeline_version: str,
    manifest: Mapping[str, Any],
    payload_json: dict[str, Any] | None,
    *,
    databricks_institution_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Build the payload object written for the versioned entrypoint."""
    out: dict[str, Any] = dict(payload_json or {})
    out["model_run_id"] = model_run_id
    out["pipeline_version"] = pipeline_version
    out["manifest"] = dict(manifest)
    if databricks_institution_name:
        out["databricks_institution_name"] = databricks_institution_name
    if model_name:
        out["model_name"] = model_name
    return out


def resolve_wheel_path(
    release_base_path: str, pipeline_version: str, wheel_name: str
) -> Path:
    """Absolute path to the wheel file under the release directory."""
    return (
        Path(release_base_path).expanduser().resolve()
        / pipeline_version
        / wheel_name
    )


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
            "MVP stable launcher: resolve model metadata, validate a versioned **runtime bundle** "
            "(manifest + optional inference_contract + compatibility), install wheel, run entrypoint."
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
        help="Override manifest entrypoint (python -m <value>).",
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
            "Require --databricks_institution_name, --model_name, and --DB_workspace "
            "(webapp passes institution + model; launcher resolves model_run_id and "
            "pipeline_version (git SHA in dev) from silver training/config.toml first, else "
            "payload_json on the pipeline_models row."
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

    manifest_path = resolve_manifest_path(args.release_base_path, pipeline_version)
    LOGGER.info("Resolved manifest path: %s", manifest_path)
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.error("Could not load manifest: %s", exc)
        return 1

    release_dir = manifest_path.parent
    effective = merge_manifest_with_optional_contract(manifest, release_dir)
    log_bundle_snapshot_presence(release_dir)

    ok_compat, compat_msg = check_runtime_bundle_compatibility(effective, spark=spark)
    if not ok_compat:
        LOGGER.error("%s", compat_msg)
        return 1
    LOGGER.info("Runtime bundle compatibility check passed.")

    cli_entrypoint = (args.entrypoint or "").strip()
    entrypoint = cli_entrypoint or str(effective.get("entrypoint") or "").strip()
    if not entrypoint:
        LOGGER.error("Entrypoint is empty after manifest/contract resolution.")
        return 1

    wheel_path = resolve_wheel_path(
        args.release_base_path,
        pipeline_version,
        str(effective["wheel"]),
    )
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
        manifest,
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
    # Avoid ``raise SystemExit(0)``: IPython / some Databricks wrappers treat any
    # SystemExit as a failed cell/job even when the code is success (0).
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
