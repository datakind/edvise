"""Resolve GenAI pipeline input paths from ``genai_active_registry.json`` on a silver volume."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from edvise.dataio.path_management import path_exists

LOGGER = logging.getLogger(__name__)

_REGISTRY_REL = ("genai_mapping", "active", "genai_active_registry.json")


def resolve_genai_pipeline_input_dir(
    silver_volume_path: str,
) -> str:
    """
    Resolve the GenAI pipeline input directory under a silver volume.

    Preferred (current) layout::

        {silver}/genai_mapping/runs/onboard/{onboard_run_id}/pipeline_input

    The ``onboard_run_id`` is read from ``genai_mapping/active/genai_active_registry.json``
    under the silver root.
    """
    silver_root = silver_volume_path.rstrip("/").rstrip(os.sep)
    registry_path = os.path.join(silver_root, *_REGISTRY_REL)
    if not path_exists(registry_path):
        raise FileNotFoundError(
            f"GenAI active registry not found at {registry_path!r}. "
            "Expected genai_mapping/active/genai_active_registry.json under the silver volume."
        )
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    run_id = payload.get("onboard_run_id")
    if not run_id or not isinstance(run_id, str):
        raise ValueError(
            f"Registry {registry_path!r} must contain a non-empty string "
            f"'onboard_run_id'; got {run_id!r}."
        )

    preferred = os.path.join(
        silver_root,
        "genai_mapping",
        "runs",
        "onboard",
        run_id,
        "pipeline_input",
    )
    if not path_exists(preferred):
        raise FileNotFoundError(
            f"GenAI pipeline_input dir not found at {preferred!r} (onboard_run_id={run_id!r})."
        )
    LOGGER.info(
        "Resolved GenAI pipeline_input dir (onboard_run_id=%r) -> %s", run_id, preferred
    )
    return preferred
