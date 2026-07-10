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


def resolve_model_run_id_from_uc_registry(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    model_name: str,
    logger: logging.Logger = LOGGER,
) -> str | None:
    """
    Resolve ``model_run_id`` from Unity Catalog registered model (same as PDP ingestion).

    Used when ``pipeline_models`` has no row but the model is registered in UC gold.
    """
    try:
        from edvise.utils.databricks import get_latest_uc_model_run_id

        run_id = get_latest_uc_model_run_id(
            model_name=model_name,
            workspace=db_workspace,
            institution=databricks_institution_name,
        )
    except Exception as exc:
        logger.warning(
            "Could not resolve model_run_id from UC registry for institution_id=%r "
            "model_name=%r: %s",
            databricks_institution_name,
            model_name,
            exc,
        )
        return None
    rid = str(run_id).strip() if run_id else ""
    if not rid:
        return None
    logger.info(
        "Resolved model_run_id from UC model registry (%s.%s_gold.%s): %s",
        db_workspace,
        databricks_institution_name,
        model_name,
        rid,
    )
    return rid


def resolve_pipeline_version_for_model_run(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    model_run_id: str,
    payload_json: str | None = None,
    logger: logging.Logger = LOGGER,
) -> str | None:
    """Read ``pipeline_version`` from silver training config, then optional payload_json."""
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
        pv = pipeline_version_from_payload_json_str(payload_json)
        if pv:
            logger.info("pipeline_version from pipeline_models.payload_json: %s", pv)
    return pv


def resolve_model_run_and_pipeline_version(
    *,
    spark: SparkSQL,
    db_workspace: str,
    databricks_institution_name: str,
    model_name: str,
    model_run_id_override: str | None = None,
    logger: logging.Logger = LOGGER,
) -> tuple[str, str] | None:
    """
    Resolve ``(model_run_id, pipeline_version)`` for versioned inference.

    ``model_run_id`` resolution order:

    1. Explicit ``model_run_id_override`` (job parameter)
    2. Latest ``pipeline_models`` row for institution + model_name
    3. Unity Catalog registered model (latest version run_id)

    ``pipeline_version`` resolution order (after ``model_run_id`` is known):

    1. Silver ``.../training/config.toml``
    2. ``pipeline_models.payload_json`` (only when step 2 above succeeded)
    """
    payload_json: str | None = None
    model_run_id: str | None = None

    override = (model_run_id_override or "").strip()
    if override:
        model_run_id = override
        logger.info("Using explicit model_run_id override: %s", model_run_id)
    else:
        q = sql_select_latest_pipeline_model(
            db_workspace, databricks_institution_name, model_name
        )
        logger.info("pipeline_models lookup:\n%s", q)
        rows = spark.sql(q).collect()
        if rows:
            row = rows[0]
            model_run_id = str(row["model_run_id"]).strip()
            if not model_run_id:
                logger.error("pipeline_models row has empty model_run_id")
                return None
            raw_payload = row["payload_json"]
            payload_json = str(raw_payload) if raw_payload is not None else None
        else:
            logger.info(
                "No pipeline_models row for institution_id=%r model_name=%r in "
                "%s.default; trying UC model registry",
                databricks_institution_name,
                model_name,
                db_workspace,
            )
            model_run_id = resolve_model_run_id_from_uc_registry(
                db_workspace=db_workspace,
                databricks_institution_name=databricks_institution_name,
                model_name=model_name,
                logger=logger,
            )

    if not model_run_id:
        logger.error(
            "Could not resolve model_run_id for institution_id=%r model_name=%r "
            "(pipeline_models, UC registry, and model_run_id override all failed)",
            databricks_institution_name,
            model_name,
        )
        return None

    pipeline_version = resolve_pipeline_version_for_model_run(
        db_workspace=db_workspace,
        databricks_institution_name=databricks_institution_name,
        model_run_id=model_run_id,
        payload_json=payload_json,
        logger=logger,
    )
    if not pipeline_version:
        cfg_path = silver_training_config_path(
            db_workspace, databricks_institution_name, model_run_id
        )
        logger.error(
            "Could not resolve pipeline_version from %s or payload_json",
            cfg_path,
        )
        return None

    logger.info(
        "Resolved model_run_id=%s pipeline_version=%s",
        model_run_id,
        pipeline_version,
    )
    return model_run_id, pipeline_version


def resolve_release_dir(release_base_path: str, pipeline_version: str) -> Path:
    from pipelines.pdp.launchers.pipeline_version_ref import sanitize_release_dir_name

    segment = sanitize_release_dir_name(pipeline_version)
    return Path(release_base_path).expanduser().resolve() / segment
