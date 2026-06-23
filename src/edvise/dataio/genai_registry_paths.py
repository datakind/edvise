"""Resolve GenAI pipeline input paths from ``genai_active_registry.json`` on a silver volume."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from edvise.dataio.path_management import path_exists

LOGGER = logging.getLogger(__name__)

_REGISTRY_REL = ("genai_mapping", "active", "genai_active_registry.json")


def _pipeline_input_dir(silver_root: str, *, mode: str, run_id: str) -> str:
    return os.path.join(
        silver_root,
        "genai_mapping",
        "runs",
        mode,
        run_id.strip(),
        "pipeline_input",
    )


def resolve_genai_pipeline_input_dir(
    silver_volume_path: str,
    *,
    job_type: str,
) -> str:
    """
    Resolve the GenAI pipeline input directory under a silver volume.

    Reads ``genai_mapping/active/genai_active_registry.json`` and selects the run tree
    based on ES job mode:

    * ``job_type="training"`` → ``runs/onboard/{onboard_run_id}/pipeline_input``
      (onboard gate_2 snapshot; stable holdout for model development).
    * ``job_type="inference"`` → ``runs/execute/{execute_run_id}/pipeline_input``
      (latest recurring execute output; requires ``execute_run_id`` in the registry).
    """
    mode = str(job_type).strip().lower()
    if mode not in ("training", "inference"):
        raise ValueError(
            f"job_type must be 'training' or 'inference' for GenAI input resolution; got {job_type!r}."
        )

    silver_root = silver_volume_path.rstrip("/").rstrip(os.sep)
    registry_path = os.path.join(silver_root, *_REGISTRY_REL)
    if not path_exists(registry_path):
        raise FileNotFoundError(
            f"GenAI active registry not found at {registry_path!r}. "
            "Expected genai_mapping/active/genai_active_registry.json under the silver volume."
        )
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    onboard_run_id = payload.get("onboard_run_id")
    if not onboard_run_id or not isinstance(onboard_run_id, str):
        raise ValueError(
            f"Registry {registry_path!r} must contain a non-empty string "
            f"'onboard_run_id'; got {onboard_run_id!r}."
        )

    if mode == "training":
        onboard_path = _pipeline_input_dir(
            silver_root, mode="onboard", run_id=onboard_run_id
        )
        if not path_exists(onboard_path):
            raise FileNotFoundError(
                f"GenAI pipeline_input dir not found at {onboard_path!r} "
                f"(onboard_run_id={onboard_run_id!r})."
            )
        LOGGER.info(
            "Resolved GenAI pipeline_input dir for training (onboard_run_id=%r) -> %s",
            onboard_run_id,
            onboard_path,
        )
        return onboard_path

    execute_run_id = payload.get("execute_run_id")
    if not execute_run_id or not isinstance(execute_run_id, str):
        raise FileNotFoundError(
            f"GenAI active registry at {registry_path!r} has no execute_run_id; "
            "inference requires a completed genai mapping execute run."
        )
    execute_path = _pipeline_input_dir(
        silver_root, mode="execute", run_id=execute_run_id
    )
    if not path_exists(execute_path):
        raise FileNotFoundError(
            f"GenAI pipeline_input dir not found at {execute_path!r} "
            f"(execute_run_id={execute_run_id!r}). "
            "Run genai mapping execute before ES inference."
        )
    LOGGER.info(
        "Resolved GenAI pipeline_input dir for inference (execute_run_id=%r) -> %s",
        execute_run_id,
        execute_path,
    )
    return execute_path
