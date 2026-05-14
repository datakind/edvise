#!/usr/bin/env python3
"""
MVP launcher: install a versioned Edvise wheel from a release manifest, then run a
versioned entrypoint. Does not import ``edvise`` before the wheel install (avoids
accidentally using the Git checkout on the driver).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol

LOGGER = logging.getLogger("versioned_inference_launcher")

DEFAULT_RELEASE_BASE = "/Volumes/dev_sst_02/default/edvise_releases"


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
    """Read ``pipeline_version`` from the ``pipeline_models.payload_json`` string."""
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
    """Parse ``pipeline_version`` from a PDP ``config.toml`` (top-level key)."""
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

    1. Silver ``training/config.toml`` for that run (authoritative training config).
    2. Else ``payload_json.pipeline_version`` on the model row (git SHA recorded at train time).
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
                "pipeline_version from pipeline_models.payload_json (git SHA): %s",
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
    """Return ``<release_base_path>/<pipeline_version>/manifest.json``."""
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
        description="MVP: install versioned Edvise wheel and run manifest entrypoint.",
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
            "pipeline_version from silver training/config.toml first, else "
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

    cli_entrypoint = (args.entrypoint or "").strip()
    entrypoint = cli_entrypoint or str(manifest["entrypoint"]).strip()
    if not entrypoint:
        LOGGER.error("Entrypoint is empty after CLI/manifest resolution.")
        return 1

    wheel_path = resolve_wheel_path(
        args.release_base_path,
        pipeline_version,
        str(manifest["wheel"]),
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
    raise SystemExit(main())
