"""Resolve model_run_id and archived pipeline_version from pipeline_models + silver config."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from edvise.runtime.versioned_inference.pipeline_version_ref import (
    sanitize_release_dir_name,
)

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
        "SELECT model_run_id "
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


def pipeline_version_from_config_toml(text: str) -> str | None:
    try:
        data = tomllib.loads(text)
    except Exception:
        return None
    if isinstance(data, dict):
        v = data.get("pipeline_version")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def get_spark_session() -> Any:
    """Return active Spark session on Databricks; ``None`` when unavailable."""
    from edvise.utils.databricks import get_spark_session_or_none

    return get_spark_session_or_none()


def resolve_model_run_id_from_uc_registry(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    model_name: str,
    logger: logging.Logger = LOGGER,
) -> str | None:
    """Resolve ``model_run_id`` from Unity Catalog when ``pipeline_models`` has no row."""
    try:
        from edvise.utils.databricks import get_latest_uc_model_run_id

        run_id = get_latest_uc_model_run_id(
            model_name,
            db_workspace,
            databricks_institution_name,
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
    rid = str(run_id).strip()
    if not rid:
        return None
    logger.info(
        "Resolved model_run_id from UC model registry (%s.%s_gold): %s",
        db_workspace,
        databricks_institution_name,
        rid,
    )
    return rid


def resolve_archived_pipeline_version(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    model_run_id: str,
    logger: logging.Logger = LOGGER,
) -> str | None:
    """Read archived ``pipeline_version`` from silver training ``config.toml``."""
    cfg_path = silver_training_config_path(
        db_workspace, databricks_institution_name, model_run_id
    )
    if not cfg_path.is_file():
        logger.error("config.toml not found at %s", cfg_path)
        return None
    try:
        pv = pipeline_version_from_config_toml(cfg_path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Failed to read config.toml: %s", exc)
        return None
    if pv:
        logger.info(
            "archived_pipeline_version from silver training config.toml (%s): %s",
            cfg_path,
            pv,
        )
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
    Resolve ``(model_run_id, archived_pipeline_version)`` for versioned inference.

    ``model_run_id`` resolution order:

    1. Explicit ``model_run_id_override`` (job parameter)
    2. Latest ``pipeline_models`` row for institution + model_name
    3. Unity Catalog registered model (latest version run_id)

    ``archived_pipeline_version`` is read from silver ``.../training/config.toml``.
    """
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

    archived_pipeline_version = resolve_archived_pipeline_version(
        db_workspace=db_workspace,
        databricks_institution_name=databricks_institution_name,
        model_run_id=model_run_id,
        logger=logger,
    )
    if not archived_pipeline_version:
        cfg_path = silver_training_config_path(
            db_workspace, databricks_institution_name, model_run_id
        )
        logger.error("Could not resolve archived_pipeline_version from %s", cfg_path)
        return None

    logger.info(
        "Resolved model_run_id=%s archived_pipeline_version=%s",
        model_run_id,
        archived_pipeline_version,
    )
    return model_run_id, archived_pipeline_version


def resolve_release_dir(release_base_path: str, pipeline_version: str) -> Path:
    segment = sanitize_release_dir_name(pipeline_version)
    return Path(release_base_path).expanduser().resolve() / segment
