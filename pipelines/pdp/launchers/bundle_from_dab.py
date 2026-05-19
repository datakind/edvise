"""
Parse archived Databricks bundle YAML into runtime metadata for the launcher.

The release directory holds ``databricks_bundle_snapshot/`` (copied at publish time from
Git at ``pipeline_version``) plus a minimal ``release.json`` (wheel name / entrypoint only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_ENTRYPOINT = "edvise.runtime.inference_driver"
DEFAULT_INFERENCE_JOB_KEY = "edvise_github_sourced_pdp_inference_pipeline"
DEFAULT_INFERENCE_YML = (
    "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
)

# DBR 15.4 ML images on Databricks use Python 3.11 (launcher compatibility hint).
_DBR_PYTHON_HINTS: dict[str, str] = {
    "15.4": "3.11",
    "14.3": "3.10",
    "13.3": "3.10",
}


def release_json_path(release_dir: Path) -> Path:
    return release_dir / "release.json"


def inference_yml_path(release_dir: Path, relative: str = DEFAULT_INFERENCE_YML) -> Path:
    return release_dir / relative


def write_release_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def load_minimal_release_json(path: Path) -> dict[str, Any]:
    """Load optional ``release.json`` (wheel / entrypoint / job key overrides only)."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        msg = f"{path} must be a JSON object"
        raise TypeError(msg)
    return data


def discover_wheel_filename(release_dir: Path, hint: str | None = None) -> str | None:
    """Return wheel basename from hint or the sole ``*.whl`` in ``release_dir``."""
    if hint and str(hint).strip():
        name = str(hint).strip()
        if (release_dir / name).is_file():
            return name
    wheels = sorted(release_dir.glob("*.whl"))
    if len(wheels) == 1:
        return wheels[0].name
    if len(wheels) > 1:
        LOGGER.warning(
            "Multiple wheels in %s; set wheel in release.json. Found: %s",
            release_dir,
            [w.name for w in wheels],
        )
    return None


def _python_hint_for_dbr(dbr: str | None) -> str | None:
    if not dbr:
        return None
    for prefix, py in _DBR_PYTHON_HINTS.items():
        if dbr.strip().startswith(prefix):
            return py
    return None


def _collect_pypi_packages(tasks: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    if not isinstance(tasks, list):
        return out
    for task in tasks:
        if not isinstance(task, dict):
            continue
        libs = task.get("libraries")
        if not isinstance(libs, list):
            continue
        for lib in libs:
            if not isinstance(lib, dict):
                continue
            pypi = lib.get("pypi")
            if isinstance(pypi, dict):
                pkg = pypi.get("package")
                if isinstance(pkg, str) and pkg.strip() and pkg not in seen:
                    seen.add(pkg)
                    out.append(pkg.strip())
    return out


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


def _collect_job_parameter_names(parameters: list[Any]) -> list[str]:
    names: list[str] = []
    if not isinstance(parameters, list):
        return names
    for param in parameters:
        if isinstance(param, dict):
            name = param.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


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
    """
    Parse ``github_pdp_inference.yml`` (or compatible) into launcher metadata.

    Returns keys: ``job_key``, ``job_name``, ``expected_steps``, ``job_parameters``,
    ``pypi_packages``, ``required_runtime``, ``inference_yml_path``.
    """
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

    tasks = job.get("tasks")
    task_keys = _collect_task_keys(tasks if isinstance(tasks, list) else [])
    job_params = _collect_job_parameter_names(
        job.get("parameters") if isinstance(job.get("parameters"), list) else []
    )
    pypi = _collect_pypi_packages(tasks if isinstance(tasks, list) else [])
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
        "job_parameters": job_params,
        "pypi_packages": pypi,
        "required_runtime": required_runtime,
        "inference_yml_path": str(yml_path),
        "execution_mode": "wheel",
    }


def build_effective_release(
    release_dir: Path,
    pipeline_version: str,
    *,
    inference_yml_relative: str = DEFAULT_INFERENCE_YML,
    inference_job_key: str = DEFAULT_INFERENCE_JOB_KEY,
    logger: logging.Logger = LOGGER,
) -> dict[str, Any]:
    """
    Combine minimal ``release.json`` with metadata parsed from the archived inference YAML.

    ``release.json`` may only specify ``wheel``, ``entrypoint``, ``inference_job_key``,
    ``execution_mode``, or partial ``required_runtime`` overrides.
    """
    release_dir = release_dir.expanduser().resolve()
    yml_path = inference_yml_path(release_dir, inference_yml_relative)
    parsed = parse_inference_job_from_yaml(
        yml_path, job_key=inference_job_key
    )
    overrides = load_minimal_release_json(release_json_path(release_dir))

    effective: dict[str, Any] = dict(parsed)
    effective["pipeline_version"] = pipeline_version
    effective["git_sha"] = pipeline_version
    effective["bundle_snapshot_dir"] = str(release_dir / "databricks_bundle_snapshot")

    wheel_hint = overrides.get("wheel")
    wheel = discover_wheel_filename(release_dir, str(wheel_hint) if wheel_hint else None)
    if not wheel:
        msg = (
            f"No wheel in {release_dir} (add release.json with wheel or a single *.whl)"
        )
        raise FileNotFoundError(msg)
    effective["wheel"] = wheel

    entrypoint = overrides.get("entrypoint")
    effective["entrypoint"] = (
        str(entrypoint).strip()
        if isinstance(entrypoint, str) and entrypoint.strip()
        else DEFAULT_ENTRYPOINT
    )

    if isinstance(overrides.get("execution_mode"), str):
        effective["execution_mode"] = overrides["execution_mode"].strip()

    rr_override = overrides.get("required_runtime")
    if isinstance(rr_override, dict):
        rr = dict(effective.get("required_runtime") or {})
        rr.update({k: str(v) for k, v in rr_override.items() if v is not None})
        effective["required_runtime"] = rr

    # Launcher payload must include these; job_parameters documents DAB contract separately.
    effective["required_payload_fields"] = [
        "model_run_id",
        "pipeline_version",
        "release",
    ]

    logger.info(
        "Built release metadata from %s: job=%s steps=%s dbr=%s wheel=%s",
        yml_path.name,
        effective.get("job_name"),
        effective.get("expected_steps"),
        (effective.get("required_runtime") or {}).get("databricks_runtime"),
        wheel,
    )
    return effective


def write_release_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
