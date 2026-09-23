"""Split archived ES / GenAI jobs into dual-pin submit segments."""

from __future__ import annotations

import copy
import re
from typing import Any

# Tasks that cannot be submitted via runs/submit without bundle resolution / compute.
_NON_SUBMIT_TASK_MARKERS = ("condition_task", "run_job_task")

_TASK_VALUE_REF = re.compile(
    r"\{\{\s*tasks\.([A-Za-z0-9_]+)\.values\.([A-Za-z0-9_]+)\s*\}\}"
)

DATA_INGESTION_TASK_KEY = "data_ingestion"
DATA_AUDIT_TASK_KEY = "data_audit"

# Published by es_gcp_databricks_ingestion via dbutils.jobs.taskValues.
INGESTION_HANDOFF_KEYS: tuple[str, ...] = (
    "bronze_batch_dir",
    "config_file_path",
    "cohort_dataset_validated_path",
    "course_dataset_validated_path",
)


def task_key_of(task: dict[str, Any]) -> str:
    key = task.get("task_key")
    return key.strip() if isinstance(key, str) else ""


def is_submitable_spark_task(task: dict[str, Any]) -> bool:
    """True when the task has executable compute payload (not condition / run_job)."""
    if not isinstance(task, dict):
        return False
    if any(marker in task for marker in _NON_SUBMIT_TASK_MARKERS):
        return False
    return any(
        key in task
        for key in (
            "spark_python_task",
            "notebook_task",
            "python_wheel_task",
            "spark_jar_task",
            "sql_task",
            "dbt_task",
        )
    )


def filter_depends_on(
    task: dict[str, Any], allowed_task_keys: set[str]
) -> dict[str, Any]:
    """Drop depends_on edges that point outside ``allowed_task_keys``."""
    out = copy.deepcopy(task)
    deps = out.get("depends_on")
    if not isinstance(deps, list):
        return out
    kept: list[Any] = []
    for dep in deps:
        if not isinstance(dep, dict):
            continue
        key = dep.get("task_key")
        if isinstance(key, str) and key.strip() in allowed_task_keys:
            kept.append(copy.deepcopy(dep))
    if kept:
        out["depends_on"] = kept
    else:
        out.pop("depends_on", None)
    return out


def select_tasks_by_keys(
    tasks: list[Any],
    keep_keys: set[str],
) -> list[dict[str, Any]]:
    """Return deep-copied tasks whose keys are in ``keep_keys``, deps filtered."""
    selected: list[dict[str, Any]] = []
    for raw in tasks:
        if not isinstance(raw, dict):
            continue
        key = task_key_of(raw)
        if key not in keep_keys:
            continue
        selected.append(filter_depends_on(raw, keep_keys))
    return selected


def spark_task_keys_in_order(tasks: list[Any]) -> list[str]:
    """Ordered task_keys for submitable spark-like tasks only."""
    keys: list[str] = []
    for raw in tasks:
        if isinstance(raw, dict) and is_submitable_spark_task(raw):
            key = task_key_of(raw)
            if key:
                keys.append(key)
    return keys


def es_full_task_keys(tasks: list[Any]) -> set[str]:
    """All spark tasks (no condition / run_job) for a single non-GenAI ES child run."""
    return set(spark_task_keys_in_order(tasks))


def es_ingestion_task_keys() -> set[str]:
    """The ES data-ingestion child run used before GenAI execute."""
    return {DATA_INGESTION_TASK_KEY}


def es_inference_task_keys(tasks: list[Any]) -> set[str]:
    """The ES inference child run: data_audit through output_publish."""
    ordered = spark_task_keys_in_order(tasks)
    if DATA_AUDIT_TASK_KEY not in ordered:
        # Older YAMLs without the GenAI branch may be linear from data_ingestion.
        return {k for k in ordered if k != DATA_INGESTION_TASK_KEY}
    start = ordered.index(DATA_AUDIT_TASK_KEY)
    return set(ordered[start:])


def job_with_selected_tasks(
    job: dict[str, Any],
    keep_keys: set[str],
) -> dict[str, Any]:
    """Copy ``job`` keeping only selected tasks (deps rewritten)."""
    out = copy.deepcopy(job)
    raw_tasks = out.get("tasks")
    tasks = raw_tasks if isinstance(raw_tasks, list) else []
    out["tasks"] = select_tasks_by_keys(tasks, keep_keys)
    if not out["tasks"]:
        msg = f"No tasks left after selecting keys={sorted(keep_keys)}"
        raise ValueError(msg)
    return out


def replace_task_value_refs(obj: Any, handoff: dict[str, str]) -> Any:
    """
    Replace ``{{tasks.data_ingestion.values.<key>}}`` with handoff literals.

    Only ``data_ingestion`` values are substituted (dual-pin handoff). Unknown
    keys become empty strings so submit does not leave unresolved templates.
    """

    def _sub_string(text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            task_key = match.group(1)
            value_key = match.group(2)
            if task_key != DATA_INGESTION_TASK_KEY:
                return match.group(0)
            return str(handoff.get(value_key, ""))

        return _TASK_VALUE_REF.sub(repl, text)

    if isinstance(obj, str):
        return _sub_string(obj)
    if isinstance(obj, list):
        return [replace_task_value_refs(item, handoff) for item in obj]
    if isinstance(obj, dict):
        return {k: replace_task_value_refs(v, handoff) for k, v in obj.items()}
    return obj


def apply_ingestion_handoff_to_job(
    job: dict[str, Any], handoff: dict[str, str]
) -> dict[str, Any]:
    """Deep-copy job and rewrite data_ingestion task-value refs to literals."""
    out = copy.deepcopy(job)
    out["tasks"] = replace_task_value_refs(out.get("tasks") or [], handoff)
    return out
