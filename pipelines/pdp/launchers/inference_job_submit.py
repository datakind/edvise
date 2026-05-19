"""
Submit a versioned multi-task inference run from archived ``github_pdp_inference.yml``.

Uses the Databricks Jobs API ``runs/submit`` with ``git_source`` pinned to
``pipeline_version`` (training git SHA). This is the supported way to run the full
inference DAG with per-task logs — not by mutating the MVP job graph at runtime.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

from pipelines.pdp.launchers.bundle_from_dab import (
    DEFAULT_INFERENCE_JOB_KEY,
    DEFAULT_INFERENCE_YML,
    inference_yml_path,
    load_inference_job_definition,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_GIT_URL = "https://github.com/datakind/edvise"
_BUNDLE_VAR = re.compile(r"\$\{var\.[^}]+\}")


def _contains_bundle_var(value: Any) -> bool:
    return isinstance(value, str) and _BUNDLE_VAR.search(value) is not None


def _strip_unresolved_bundle_refs(obj: Any) -> Any:
    """Remove dict entries whose string values contain unresolved ``${var.*}``."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, val in obj.items():
            if _contains_bundle_var(val):
                continue
            cleaned = _strip_unresolved_bundle_refs(val)
            if cleaned is None:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            out[key] = cleaned
        return out
    if isinstance(obj, list):
        return [_strip_unresolved_bundle_refs(item) for item in obj]
    return obj


def resolve_job_parameter_specs(
    parameter_specs: list[Any],
    overrides: dict[str, str],
) -> list[dict[str, str]]:
    """Build Jobs API parameter list; apply overrides; drop unresolved ``${var}``."""
    resolved: list[dict[str, str]] = []
    if not isinstance(parameter_specs, list):
        return resolved
    for spec in parameter_specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        default = spec.get("default", "")
        if _contains_bundle_var(default):
            default = overrides.get(name, "")
        if name in overrides and overrides[name] is not None:
            default = overrides[name]
        resolved.append({"name": name, "default": str(default)})
    return resolved


def _job_cluster_map(job_clusters: list[Any]) -> dict[str, dict[str, Any]]:
    """Map ``job_cluster_key`` → ``new_cluster`` spec from bundle job YAML."""
    out: dict[str, dict[str, Any]] = {}
    for entry in job_clusters:
        if not isinstance(entry, dict):
            continue
        key = entry.get("job_cluster_key")
        new_cluster = entry.get("new_cluster")
        if isinstance(key, str) and key.strip() and isinstance(new_cluster, dict):
            out[key.strip()] = copy.deepcopy(new_cluster)
    return out


def inline_job_clusters_for_submit(
    tasks: list[Any],
    job_clusters: list[Any],
    *,
    logger: logging.Logger = LOGGER,
) -> list[dict[str, Any]]:
    """
    ``runs/submit`` does not support shared ``job_clusters``; attach ``new_cluster`` per task.

    DAB-deployed jobs use ``job_cluster_key`` + ``job_clusters``; submit requires inline clusters.
    """
    cluster_map = _job_cluster_map(job_clusters)
    if not cluster_map:
        msg = "No job_clusters definitions found for runs/submit"
        raise ValueError(msg)

    submit_tasks: list[dict[str, Any]] = []
    for raw in tasks:
        if not isinstance(raw, dict):
            continue
        task = copy.deepcopy(raw)
        key = task.pop("job_cluster_key", None)
        if isinstance(key, str) and key.strip():
            cluster_key = key.strip()
            if cluster_key not in cluster_map:
                msg = f"Task {task.get('task_key')!r} references unknown job_cluster_key={cluster_key!r}"
                raise ValueError(msg)
            task["new_cluster"] = copy.deepcopy(cluster_map[cluster_key])
        elif "new_cluster" not in task:
            msg = f"Task {task.get('task_key')!r} has no job_cluster_key or new_cluster"
            raise ValueError(msg)
        submit_tasks.append(task)

    if not submit_tasks:
        msg = "No tasks after inlining job clusters for runs/submit"
        raise ValueError(msg)
    logger.info(
        "Inlined %s job cluster(s) onto %s submit task(s) (runs/submit API)",
        len(cluster_map),
        len(submit_tasks),
    )
    return submit_tasks


def build_submit_run_body(
    job: dict[str, Any],
    *,
    pipeline_version: str,
    git_url: str,
    run_name: str,
    parameter_overrides: dict[str, str],
    inference_job_key: str = DEFAULT_INFERENCE_JOB_KEY,
) -> dict[str, Any]:
    """
    Convert archived inference job YAML to a ``jobs/runs/submit`` request body.

    Strips bundle-only sections (permissions, unresolved webhooks) and pins git.
    """
    job = copy.deepcopy(job)
    tasks = job.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        msg = f"Job {inference_job_key!r} has no tasks"
        raise ValueError(msg)

    job_clusters = job.get("job_clusters")
    if not isinstance(job_clusters, list):
        msg = f"Job {inference_job_key!r} has no job_clusters"
        raise ValueError(msg)

    parameters = resolve_job_parameter_specs(
        job.get("parameters") if isinstance(job.get("parameters"), list) else [],
        parameter_overrides,
    )

    cleaned_tasks = _strip_unresolved_bundle_refs(tasks)
    cleaned_clusters = _strip_unresolved_bundle_refs(job_clusters)
    if not isinstance(cleaned_tasks, list):
        msg = f"Job {inference_job_key!r} tasks could not be sanitized for submit"
        raise TypeError(msg)
    if not isinstance(cleaned_clusters, list):
        msg = f"Job {inference_job_key!r} job_clusters could not be sanitized for submit"
        raise TypeError(msg)
    submit_tasks = inline_job_clusters_for_submit(cleaned_tasks, cleaned_clusters)

    body: dict[str, Any] = {
        "run_name": run_name,
        "git_source": {
            "git_url": git_url.rstrip("/"),
            "git_provider": "gitHub",
            "git_commit": pipeline_version,
        },
        "tasks": submit_tasks,
    }
    if parameters:
        body["parameters"] = parameters

    queue = job.get("queue")
    if isinstance(queue, dict) and queue.get("enabled") is not None:
        body["queue"] = queue

    email = job.get("email_notifications")
    if isinstance(email, dict) and not _contains_bundle_var(str(email)):
        body["email_notifications"] = email

    run_as = parameter_overrides.get("ds_run_as", "").strip()
    if run_as:
        body["run_as"] = {"service_principal_name": run_as}

    return body


def submit_inference_run(
    submit_body: dict[str, Any],
    *,
    dry_run: bool = False,
    workspace_client: Any | None = None,
    logger: logging.Logger = LOGGER,
) -> int:
    """POST ``/api/2.1/jobs/runs/submit``; return ``run_id`` (0 if ``dry_run``)."""
    if dry_run:
        logger.info(
            "dry-run submit (%s tasks): %s",
            len(submit_body.get("tasks") or []),
            json.dumps(submit_body, indent=2, default=str)[:8000],
        )
        return 0

    if workspace_client is None:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    response = workspace_client.api_client.do(
        "POST",
        "/api/2.1/jobs/runs/submit",
        body=submit_body,
    )
    if not isinstance(response, dict) or "run_id" not in response:
        msg = f"Unexpected submit response: {response!r}"
        raise RuntimeError(msg)
    run_id = int(response["run_id"])
    host = getattr(getattr(workspace_client, "config", None), "host", None)
    if host:
        logger.info(
            "Submitted versioned inference run_id=%s — monitor at %s/#job/%s",
            run_id,
            host.rstrip("/"),
            run_id,
        )
    else:
        logger.info("Submitted versioned inference run_id=%s", run_id)
    return run_id


def submit_versioned_inference_from_bundle(
    release_dir: Path,
    *,
    pipeline_version: str,
    parameter_overrides: dict[str, str],
    git_url: str = DEFAULT_GIT_URL,
    inference_job_key: str = DEFAULT_INFERENCE_JOB_KEY,
    inference_yml_relative: str = DEFAULT_INFERENCE_YML,
    dry_run: bool = False,
    workspace_client: Any | None = None,
    logger: logging.Logger = LOGGER,
) -> int:
    """Load archived inference YAML from ``release_dir`` and submit a multi-task run."""
    yml_path = inference_yml_path(release_dir, inference_yml_relative)
    job = load_inference_job_definition(yml_path, job_key=inference_job_key)
    inst = parameter_overrides.get("databricks_institution_name", "unknown")
    model = parameter_overrides.get("model_name", "unknown")
    run_name = (
        f"versioned-inference-{inst}-{model}-{pipeline_version[:12]}"
    )
    body = build_submit_run_body(
        job,
        pipeline_version=pipeline_version,
        git_url=git_url,
        run_name=run_name,
        parameter_overrides=parameter_overrides,
        inference_job_key=inference_job_key,
    )
    logger.info(
        "Submitting inference job %r at git commit %s (%s tasks)",
        job.get("name", inference_job_key),
        pipeline_version,
        len(body.get("tasks") or []),
    )
    return submit_inference_run(
        body,
        dry_run=dry_run,
        workspace_client=workspace_client,
        logger=logger,
    )
