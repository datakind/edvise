"""Resolve model_run_id and pipeline_version from pipeline_models + silver config."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)


class SparkSQL(Protocol):
    def sql(self, query: str) -> Any: ...


def escape_sql_string_literal(value: str) -> str:
    return value.replace("'", "''")


def sql_select_latest_pipeline_model(
    db_workspace: str, institution_id: str, model_name: str
) -> str:
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
    return Path(
        f"/Volumes/{db_workspace}/{databricks_institution_name}_silver/"
        f"silver_volume/{model_run_id}/training/config.toml"
    )


def pipeline_version_from_payload_json_str(raw: str | None) -> str | None:
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
    try:
        from pyspark.sql import SparkSession  # type: ignore[import-not-found]

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as exc:
        LOGGER.warning("Could not obtain SparkSession: %s", exc)
        return None


def resolve_model_run_and_pipeline_version(
    *,
    spark: SparkSQL,
    db_workspace: str,
    databricks_institution_name: str,
    model_name: str,
    logger: logging.Logger = LOGGER,
) -> tuple[str, str] | None:
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
        pv = pipeline_version_from_payload_json_str(row["payload_json"])
        if pv:
            logger.info("pipeline_version from pipeline_models.payload_json: %s", pv)

    if not pv:
        logger.error(
            "Could not resolve pipeline_version from %s or payload_json",
            cfg_path,
        )
        return None
    logger.info("Resolved model_run_id=%s pipeline_version=%s", model_run_id, pv)
    return model_run_id, pv


def resolve_release_dir(release_base_path: str, pipeline_version: str) -> Path:
    from pipelines.pdp.launchers.pipeline_version_ref import sanitize_release_dir_name

    segment = sanitize_release_dir_name(pipeline_version)
    return Path(release_base_path).expanduser().resolve() / segment
