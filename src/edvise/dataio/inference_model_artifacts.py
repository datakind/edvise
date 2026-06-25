"""Resolve inference artifact paths from registered Unity Catalog models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from edvise.utils.databricks import find_file_in_run_folder, get_latest_uc_model_run_id

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EsInferenceArtifacts:
    """Paths resolved from the trained model's silver run folder."""

    model_run_id: str
    config_file_path: str
    silver_run_root: str


@dataclass(frozen=True, slots=True)
class LegacyInferenceArtifacts:
    """Config and features table paths from the trained legacy model run."""

    model_run_id: str
    config_file_path: str
    features_table_path: str
    silver_run_root: str


def silver_run_root_for_model(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    model_run_id: str,
) -> str:
    return (
        f"/Volumes/{db_workspace.strip()}/"
        f"{databricks_institution_name.strip()}_silver/silver_volume/{model_run_id.strip()}"
    )


def resolve_es_inference_artifacts(
    *,
    model_name: str,
    db_workspace: str,
    databricks_institution_name: str,
) -> EsInferenceArtifacts:
    """Resolve ``config_file_path`` from the latest registered ES model version."""
    model_run_id = get_latest_uc_model_run_id(
        model_name=model_name,
        workspace=db_workspace,
        institution=databricks_institution_name,
    )
    silver_run_root = silver_run_root_for_model(
        db_workspace=db_workspace,
        databricks_institution_name=databricks_institution_name,
        model_run_id=model_run_id,
    )
    LOGGER.info("Looking for ES inference config in: %s", silver_run_root)
    config_file_path = str(
        find_file_in_run_folder(silver_run_root, keyword="config")
    )
    LOGGER.info("Using ES inference config file: %s", config_file_path)
    return EsInferenceArtifacts(
        model_run_id=model_run_id,
        config_file_path=config_file_path,
        silver_run_root=silver_run_root,
    )


def resolve_legacy_inference_artifacts(
    *,
    model_name: str,
    db_workspace: str,
    databricks_institution_name: str,
) -> LegacyInferenceArtifacts:
    """Resolve config and features table paths from the latest legacy model version."""
    model_run_id = get_latest_uc_model_run_id(
        model_name=model_name,
        workspace=db_workspace,
        institution=databricks_institution_name,
    )
    silver_run_root = silver_run_root_for_model(
        db_workspace=db_workspace,
        databricks_institution_name=databricks_institution_name,
        model_run_id=model_run_id,
    )
    LOGGER.info("Looking for legacy inference artifacts in: %s", silver_run_root)
    config_file_path = str(
        find_file_in_run_folder(silver_run_root, keyword="config")
    )
    features_table_path = str(
        find_file_in_run_folder(silver_run_root, keyword="features_table")
    )
    LOGGER.info("Using legacy config file: %s", config_file_path)
    LOGGER.info("Using legacy features table: %s", features_table_path)
    return LegacyInferenceArtifacts(
        model_run_id=model_run_id,
        config_file_path=config_file_path,
        features_table_path=features_table_path,
        silver_run_root=silver_run_root,
    )


def set_inference_config_task_value(config_file_path: str) -> None:
    """Publish ``config_file_path`` for downstream Databricks tasks when available."""
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        LOGGER.warning(
            "dbutils not available - config_file_path task value not set "
            "(expected outside Databricks)."
        )
        return

    try:
        dbutils.jobs.taskValues.set(key="config_file_path", value=config_file_path or "")
    except (AttributeError, OSError, RuntimeError, TypeError) as exc:
        LOGGER.error(
            "dbutils.jobs.taskValues.set failed for config_file_path: %s",
            exc,
            exc_info=True,
        )


def set_legacy_inference_artifact_task_values(
    *,
    config_file_path: str,
    features_table_path: str,
) -> None:
    """Publish legacy inference artifact paths for downstream tasks."""
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        LOGGER.warning(
            "dbutils not available - legacy artifact task values not set "
            "(expected outside Databricks)."
        )
        return

    try:
        dbutils.jobs.taskValues.set(key="config_file_path", value=config_file_path or "")
        dbutils.jobs.taskValues.set(
            key="features_table_path", value=features_table_path or ""
        )
    except (AttributeError, OSError, RuntimeError, TypeError) as exc:
        LOGGER.error(
            "dbutils.jobs.taskValues.set failed for legacy artifacts: %s",
            exc,
            exc_info=True,
        )
