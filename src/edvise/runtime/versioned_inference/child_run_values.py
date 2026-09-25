"""Read / reconstruct Databricks multi-task ``taskValues`` for ES child-run handoff."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

# Used when --dry-run builds GenAI execute / ES inference bodies without ingestion.
DRY_RUN_INGESTION_HANDOFF: dict[str, str] = {
    "bronze_batch_dir": "/Volumes/dry_run/placeholder/gcs_uploads/batch",
    "config_file_path": "/Volumes/dry_run/placeholder/config.toml",
    "cohort_dataset_validated_path": "",
    "course_dataset_validated_path": "",
}


def _as_mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    # SDK objects sometimes expose .as_dict()
    as_dict = getattr(raw, "as_dict", None)
    if callable(as_dict):
        data = as_dict()
        if isinstance(data, dict):
            return data
    out: dict[str, Any] = {}
    for attr in ("values", "task_values", "notebook_output"):
        if hasattr(raw, attr):
            nested = getattr(raw, attr)
            if isinstance(nested, dict):
                out.update(nested)
    return out


def extract_task_values_from_run(
    run: Any,
    *,
    task_key: str,
    logger: logging.Logger = LOGGER,
) -> dict[str, str]:
    """
    Pull string task values for ``task_key`` from a ``jobs.get_run`` response.

    Supports both dict-shaped API responses and databricks-sdk objects.
    """
    tasks = run.get("tasks") if isinstance(run, dict) else getattr(run, "tasks", None)
    if not tasks:
        raise ValueError(f"Run has no tasks; cannot read values for {task_key!r}")

    matched: Any | None = None
    for task in tasks:
        key = (
            task.get("task_key")
            if isinstance(task, dict)
            else getattr(task, "task_key", None)
        )
        if isinstance(key, str) and key.strip() == task_key:
            matched = task
            break
    if matched is None:
        raise ValueError(f"Task {task_key!r} not found on child run")

    # Prefer explicit values fields used by the Jobs API.
    candidates: list[Any] = []
    if isinstance(matched, dict):
        candidates.extend(
            [
                matched.get("values"),
                matched.get("task_values"),
                matched,
            ]
        )
    else:
        candidates.extend(
            [
                getattr(matched, "values", None),
                getattr(matched, "task_values", None),
                matched,
            ]
        )

    values: dict[str, str] = {}
    for candidate in candidates:
        mapping = _as_mapping(candidate)
        # Nested shape: {"task_values": {"bronze_batch_dir": "..."}}
        nested = mapping.get("task_values")
        if isinstance(nested, dict):
            mapping = {**mapping, **nested}
        for key, val in mapping.items():
            if key in {"task_key", "depends_on", "state", "run_id", "attempt_number"}:
                continue
            if val is None:
                continue
            if isinstance(val, (dict, list)):
                continue
            values[str(key)] = str(val)

    if not values:
        logger.warning(
            "No task values found on task %r; downstream segments may lack handoff paths.",
            task_key,
        )
    else:
        logger.info(
            "Read %s task value(s) from %r: %s",
            len(values),
            task_key,
            sorted(values),
        )
    return values


def require_ingestion_handoff(
    values: dict[str, str],
    *,
    required_keys: tuple[str, ...],
    hard_required: tuple[str, ...] = ("config_file_path",),
) -> dict[str, str]:
    """Return handoff map; raise if a hard-required key is missing/blank."""
    handoff = {key: str(values.get(key, "") or "") for key in required_keys}
    missing = [
        key
        for key in hard_required
        if key in handoff and not str(handoff.get(key, "")).strip()
    ]
    if missing:
        raise ValueError(
            "data_ingestion handoff missing required task value(s): "
            + ", ".join(missing)
        )
    return handoff


def reconstruct_ingestion_handoff(
    *,
    db_workspace: str,
    databricks_institution_name: str,
    model_name: str,
    batch_id: str = "",
    logger: logging.Logger = LOGGER,
) -> dict[str, str]:
    """
    Rebuild ingestion outputs without mutating the original ingestion scripts.

    Uses the same path helpers / artifact resolution the ES ingestion path uses:
    ``bronze_gcs_batch_dir`` and ``resolve_es_inference_artifacts``.
    """
    from edvise.dataio.batch_gcs_inference_ingest import bronze_gcs_batch_dir
    from edvise.dataio.inference_model_artifacts import resolve_es_inference_artifacts

    artifacts = resolve_es_inference_artifacts(
        model_name=model_name,
        db_workspace=db_workspace,
        databricks_institution_name=databricks_institution_name,
    )
    bronze = ""
    if batch_id.strip():
        bronze = bronze_gcs_batch_dir(
            db_workspace, databricks_institution_name, batch_id.strip()
        )
    handoff = {
        "bronze_batch_dir": bronze,
        "config_file_path": artifacts.config_file_path,
        "cohort_dataset_validated_path": "",
        "course_dataset_validated_path": "",
    }
    logger.info(
        "Reconstructed ingestion handoff (bronze_batch_dir=%r, config_file_path=%r)",
        handoff["bronze_batch_dir"],
        handoff["config_file_path"],
    )
    return handoff


def fetch_ingestion_handoff_from_run(
    workspace_client: Any,
    run_id: int,
    *,
    task_key: str,
    required_keys: tuple[str, ...],
    hard_required: tuple[str, ...] = ("config_file_path",),
    logger: logging.Logger = LOGGER,
) -> dict[str, str]:
    """``jobs.get_run`` then extract / require ingestion handoff keys."""
    run = workspace_client.jobs.get_run(run_id=run_id)
    values = extract_task_values_from_run(run, task_key=task_key, logger=logger)
    return require_ingestion_handoff(
        values,
        required_keys=required_keys,
        hard_required=hard_required,
    )
