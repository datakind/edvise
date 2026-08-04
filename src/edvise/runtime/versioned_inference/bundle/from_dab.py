"""
Parse archived Databricks bundle YAML for launcher validation.

The release directory holds ``databricks_bundle_snapshot/`` (YAML copied from Git at
``pipeline_version``). ``build_effective_release`` reads task keys and cluster runtime
hints from the archived inference job definition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from edvise.runtime.versioned_inference.dab_layout import default_dab_bundle_layout

LOGGER = logging.getLogger(__name__)

_DEFAULT_LAYOUT = default_dab_bundle_layout()
DEFAULT_INFERENCE_JOB_KEY = _DEFAULT_LAYOUT.inference_job_key
DEFAULT_INFERENCE_YML = _DEFAULT_LAYOUT.inference_yml_snapshot_rel

_DBR_PYTHON_HINTS: dict[str, str] = {
    "15.4": "3.11",
    "14.3": "3.10",
    "13.3": "3.10",
}


def inference_yml_path(
    release_dir: Path, relative: str = DEFAULT_INFERENCE_YML
) -> Path:
    return release_dir / relative


def _python_hint_for_dbr(dbr: str | None) -> str | None:
    if not dbr:
        return None
    for prefix, py in _DBR_PYTHON_HINTS.items():
        if dbr.strip().startswith(prefix):
            return py
    return None


def _collect_task_keys(tasks: list[Any]) -> list[str]:
    keys: list[str] = []
    if not isinstance(tasks, list):
        return keys
    for task in tasks:
        if isinstance(task, dict):
            key = task.get("task_key")
            if isinstance(key, str) and key.strip():
                keys.append(key.strip())
    return keys


def _spark_version_from_job(job: dict[str, Any]) -> str | None:
    clusters = job.get("job_clusters")
    if not isinstance(clusters, list):
        return None
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        new_cluster = cluster.get("new_cluster")
        if isinstance(new_cluster, dict):
            sv = new_cluster.get("spark_version")
            if isinstance(sv, str) and sv.strip():
                return sv.strip()
    return None


def load_inference_job_definition(
    yml_path: Path,
    *,
    job_key: str = DEFAULT_INFERENCE_JOB_KEY,
) -> dict[str, Any]:
    """Load the full inference job object from archived bundle YAML."""
    import yaml

    if not yml_path.is_file():
        msg = f"Inference bundle YAML not found: {yml_path}"
        raise FileNotFoundError(msg)
    raw = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Invalid YAML root in {yml_path}"
        raise TypeError(msg)
    resources = raw.get("resources")
    if not isinstance(resources, dict):
        msg = f"No resources section in {yml_path}"
        raise ValueError(msg)
    jobs = resources.get("jobs")
    if not isinstance(jobs, dict):
        msg = f"No resources.jobs in {yml_path}"
        raise ValueError(msg)
    job = jobs.get(job_key)
    if not isinstance(job, dict):
        available = ", ".join(sorted(jobs.keys()))
        msg = f"Job {job_key!r} not in {yml_path}; available: {available}"
        raise ValueError(msg)
    return job


def parse_inference_job_from_yaml(
    yml_path: Path,
    *,
    job_key: str = DEFAULT_INFERENCE_JOB_KEY,
) -> dict[str, Any]:
    """Parse archived inference YAML into launcher metadata for compatibility checks."""
    job = load_inference_job_definition(yml_path, job_key=job_key)
    tasks = job.get("tasks")
    task_keys = _collect_task_keys(tasks if isinstance(tasks, list) else [])
    dbr = _spark_version_from_job(job)
    required_runtime: dict[str, str] = {}
    if dbr:
        required_runtime["databricks_runtime"] = dbr
    py_hint = _python_hint_for_dbr(dbr)
    if py_hint:
        required_runtime["python"] = py_hint

    job_name = job.get("name")
    return {
        "job_key": job_key,
        "job_name": str(job_name) if job_name is not None else job_key,
        "expected_steps": task_keys,
        "required_runtime": required_runtime,
        "inference_yml_path": str(yml_path),
        "execution_mode": "git_submit",
    }


def build_effective_release(
    release_dir: Path,
    pipeline_version: str,
    *,
    inference_yml_relative: str = DEFAULT_INFERENCE_YML,
    inference_job_key: str = DEFAULT_INFERENCE_JOB_KEY,
    logger: logging.Logger = LOGGER,
) -> dict[str, Any]:
    """Build effective release metadata from archived inference YAML only."""
    release_dir = release_dir.expanduser().resolve()
    yml_path = inference_yml_path(release_dir, inference_yml_relative)
    parsed = parse_inference_job_from_yaml(yml_path, job_key=inference_job_key)

    effective: dict[str, Any] = dict(parsed)
    effective["pipeline_version"] = pipeline_version
    effective["bundle_snapshot_dir"] = str(release_dir / "databricks_bundle_snapshot")

    logger.info(
        "Built release metadata from %s: job=%s steps=%s dbr=%s",
        yml_path.name,
        effective.get("job_name"),
        effective.get("expected_steps"),
        (effective.get("required_runtime") or {}).get("databricks_runtime"),
    )
    return effective
