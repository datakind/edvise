"""Hard I/O chain checks for ES dual-pin versioned inference child runs."""

from __future__ import annotations

import logging
from pathlib import Path

from edvise.dataio.batch_gcs_inference_ingest import is_bronze_batch_ready
from edvise.dataio.genai_registry_paths import resolve_genai_pipeline_input_dir
from edvise.dataio.path_management import path_exists
from edvise.genai.mapping.shared.active_promotion import (
    GENAI_ACTIVE_REGISTRY_BASENAME,
    read_genai_active_registry,
)
from edvise.runtime.versioned_inference.genai_registry import (
    genai_active_root,
    silver_volume_root,
)

LOGGER = logging.getLogger(__name__)

# Minimum silver artifacts expected after a successful ES inference / full run.
_INFERENCE_ARTIFACT_CANDIDATES = (
    "preprocessed.parquet",
    "df_cohort_standardized.parquet",
    "df_course_standardized.parquet",
)


def assert_ingestion_outputs_ready(
    handoff: dict[str, str],
    *,
    logger: logging.Logger = LOGGER,
) -> None:
    """
    After ES ``data_ingestion``: config path exists; bronze batch ready when set.

    Raises ``FileNotFoundError`` / ``ValueError`` when the handoff cannot feed
    GenAI execute / ES inference.
    """
    config = str(handoff.get("config_file_path", "") or "").strip()
    if not config:
        raise ValueError("I/O chain: config_file_path missing after data_ingestion")
    if not path_exists(config):
        raise FileNotFoundError(
            f"I/O chain: config_file_path does not exist after data_ingestion: {config}"
        )

    bronze = str(handoff.get("bronze_batch_dir", "") or "").strip()
    if bronze:
        if not path_exists(bronze):
            raise FileNotFoundError(
                f"I/O chain: bronze_batch_dir does not exist after data_ingestion: {bronze}"
            )
        if not is_bronze_batch_ready(bronze):
            raise FileNotFoundError(
                f"I/O chain: bronze_batch_dir is not ready "
                f"(need _SUCCESS.json + data files): {bronze}"
            )
        logger.info("I/O chain OK after ingestion: bronze_batch_dir=%s", bronze)
    else:
        logger.info(
            "I/O chain OK after ingestion: config_file_path=%s (bronze_batch_dir empty)",
            config,
        )


def assert_genai_execute_outputs_ready(
    db_workspace: str,
    databricks_institution_name: str,
    *,
    bronze_batch_dir: str = "",
    logger: logging.Logger = LOGGER,
) -> str:
    """
    After GenAI execute: registry has ``execute_run_id`` and ``pipeline_input`` exists.

    Returns the resolved ``execute_run_id``.
    """
    active = genai_active_root(db_workspace, databricks_institution_name)
    registry_path = active / GENAI_ACTIVE_REGISTRY_BASENAME
    payload = read_genai_active_registry(active)
    if payload is None:
        raise FileNotFoundError(
            f"I/O chain: GenAI active registry missing after execute: {registry_path}"
        )
    execute_run_id = payload.get("execute_run_id")
    if not isinstance(execute_run_id, str) or not execute_run_id.strip():
        raise ValueError(
            f"I/O chain: registry {registry_path} has no execute_run_id after GenAI execute"
        )
    execute_run_id = execute_run_id.strip()

    silver = silver_volume_root(db_workspace, databricks_institution_name)
    pipeline_input = resolve_genai_pipeline_input_dir(str(silver), job_type="inference")
    if not path_exists(pipeline_input):
        raise FileNotFoundError(
            f"I/O chain: GenAI pipeline_input missing after execute: {pipeline_input}"
        )

    bronze = (bronze_batch_dir or "").strip()
    logger.info(
        "I/O chain OK after GenAI execute: execute_run_id=%s pipeline_input=%s "
        "bronze_batch_dir=%r",
        execute_run_id,
        pipeline_input,
        bronze,
    )
    return execute_run_id


def inference_silver_dir(
    db_workspace: str,
    databricks_institution_name: str,
    model_run_id: str,
) -> Path:
    """``…/silver_volume/{model_run_id}/inference``."""
    return (
        silver_volume_root(db_workspace, databricks_institution_name)
        / model_run_id.strip()
        / "inference"
    )


def assert_es_inference_outputs_ready(
    db_workspace: str,
    databricks_institution_name: str,
    model_run_id: str,
    *,
    db_run_id: str = "",
    logger: logging.Logger = LOGGER,
) -> Path:
    """
    After ES full / inference: at least one expected inference artifact under silver.

    ``db_run_id`` is logged for lineage (shared launcher id); artifact layout is keyed
    by ``model_run_id``.
    """
    if not model_run_id.strip():
        raise ValueError(
            "I/O chain: model_run_id required to validate ES inference outputs"
        )

    inference_dir = inference_silver_dir(
        db_workspace, databricks_institution_name, model_run_id
    )
    if not path_exists(str(inference_dir)):
        raise FileNotFoundError(
            f"I/O chain: ES inference output dir missing after inference/full: {inference_dir}"
        )

    found: list[str] = []
    for name in _INFERENCE_ARTIFACT_CANDIDATES:
        candidate = inference_dir / name
        if path_exists(str(candidate)):
            found.append(name)
    if not found:
        raise FileNotFoundError(
            f"I/O chain: no expected inference artifacts under {inference_dir} "
            f"(looked for {_INFERENCE_ARTIFACT_CANDIDATES})"
        )

    logger.info(
        "I/O chain OK after ES inference: dir=%s artifacts=%s db_run_id=%r",
        inference_dir,
        found,
        db_run_id,
    )
    return inference_dir
