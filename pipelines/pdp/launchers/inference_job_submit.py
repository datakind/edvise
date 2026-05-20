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
import time
import uuid
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
_JOB_PARAM_REF = re.compile(r"\{\{\s*job\.parameters\.([A-Za-z0-9_]+)\s*\}\}")
_JOB_RUN_ID_REF = re.compile(r"\{\{\s*job\.run_id\s*\}\}")
_TERMINAL_LIFE_CYCLE_STATES = frozenset({"TERMINATED", "SKIPPED", "INTERNAL_ERROR"})
_SUCCESS_RESULT_STATE = "SUCCESS"
_UNRESOLVED_RUN_ID = "{{job.run_id}}"


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


def job_parameter_values_from_specs(
    parameter_specs: list[dict[str, str]],
    parameter_overrides: dict[str, str],
) -> dict[str, str]:
    """Merge resolved job parameter specs with explicit overrides."""
    values = {p["name"]: p["default"] for p in parameter_specs}
    for name, val in parameter_overrides.items():
        if val is not None and str(val).strip():
            values[name] = str(val)
    return values


def ensure_concrete_db_run_id(
    parameter_values: dict[str, str],
    parameter_overrides: dict[str, str],
) -> str:
    """
    ``runs/submit`` does not assign ``{{job.run_id}}``; use override or generate one.
    """
    for source in (parameter_overrides, parameter_values):
        val = source.get("db_run_id", "")
        if val and str(val).strip() and str(val).strip() != _UNRESOLVED_RUN_ID:
            return str(val).strip()
    return f"versioned_{uuid.uuid4().hex}"


def render_job_parameter_refs(
    obj: Any,
    parameter_values: dict[str, str],
    *,
    run_id: str | None = None,
) -> Any:
    """
    Replace ``{{job.parameters.NAME}}`` (and optional ``{{job.run_id}}``) in nested structures.

    Databricks does not interpolate job parameters for one-off ``runs/submit`` payloads; tasks
    must receive literal values in ``spark_python_task.parameters`` and similar fields.
    """
    if isinstance(obj, str):

        def replace_param(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in parameter_values:
                msg = f"Missing job parameter {name!r} for runs/submit rendering"
                raise ValueError(msg)
            return parameter_values[name]

        rendered = _JOB_PARAM_REF.sub(replace_param, obj)
        if run_id is not None:
            rendered = _JOB_RUN_ID_REF.sub(run_id, rendered)
        return rendered
    if isinstance(obj, list):
        return [
            render_job_parameter_refs(item, parameter_values, run_id=run_id)
            for item in obj
        ]
    if isinstance(obj, dict):
        return {
            key: render_job_parameter_refs(val, parameter_values, run_id=run_id)
            for key, val in obj.items()
        }
    return obj


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


def build_submit_access_control_list(
    parameter_overrides: dict[str, str],
) -> list[dict[str, str]]:
    """Build ``access_control_list`` entries for ``jobs/runs/submit`` from launcher overrides."""
    acl: list[dict[str, str]] = []
    group = parameter_overrides.get("datakind_group_to_manage_workflow", "").strip()
    if group:
        acl.append(
            {
                "group_name": group,
                "permission_level": "CAN_MANAGE_RUN",
            }
        )
    viewer_user = parameter_overrides.get("viewer_user", "").strip()
    if viewer_user:
        acl.append(
            {
                "user_name": viewer_user,
                "permission_level": "CAN_VIEW",
            }
        )
    return acl


def _run_state_fields(run: Any) -> tuple[str | None, str | None]:
    state = getattr(run, "state", None)
    if state is None and isinstance(run, dict):
        state = run.get("state")
    if isinstance(state, dict):
        life = state.get("life_cycle_state")
        result = state.get("result_state")
        return (
            str(life) if life is not None else None,
            str(result) if result is not None else None,
        )
    if state is not None:
        life = getattr(state, "life_cycle_state", None)
        result = getattr(state, "result_state", None)
        return (
            str(life) if life is not None else None,
            str(result) if result is not None else None,
        )
    return None, None


def wait_for_inference_run(
    run_id: int,
    *,
    workspace_client: Any,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float | None = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """
    Poll ``jobs.get_run`` until the child run reaches a terminal state.

    Raises ``RuntimeError`` if the run does not finish with ``result_state=SUCCESS``.
    """
    start = time.monotonic()
    while True:
        run = workspace_client.jobs.get_run(run_id=run_id)
        life_cycle, result_state = _run_state_fields(run)
        logger.info(
            "Child inference run_id=%s state life_cycle=%s result=%s",
            run_id,
            life_cycle,
            result_state,
        )
        if life_cycle in _TERMINAL_LIFE_CYCLE_STATES:
            if life_cycle == "TERMINATED" and result_state == _SUCCESS_RESULT_STATE:
                monitor_url = getattr(run, "run_page_url", None)
                if monitor_url:
                    logger.info(
                        "Child inference run_id=%s succeeded — %s",
                        run_id,
                        monitor_url,
                    )
                else:
                    logger.info("Child inference run_id=%s succeeded", run_id)
                return
            msg = (
                f"Child inference run_id={run_id} finished with "
                f"life_cycle_state={life_cycle!r} result_state={result_state!r}"
            )
            raise RuntimeError(msg)
        if timeout_seconds is not None and (time.monotonic() - start) >= timeout_seconds:
            msg = (
                f"Timed out after {timeout_seconds}s waiting for child inference "
                f"run_id={run_id} (last life_cycle_state={life_cycle!r})"
            )
            raise RuntimeError(msg)
        time.sleep(poll_interval_seconds)


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
    parameter_values = job_parameter_values_from_specs(parameters, parameter_overrides)
    db_run_id = ensure_concrete_db_run_id(parameter_values, parameter_overrides)
    parameter_values["db_run_id"] = db_run_id
    for spec in parameters:
        if spec["name"] == "db_run_id":
            spec["default"] = db_run_id

    cleaned_tasks = _strip_unresolved_bundle_refs(tasks)
    cleaned_clusters = _strip_unresolved_bundle_refs(job_clusters)
    if not isinstance(cleaned_tasks, list):
        msg = f"Job {inference_job_key!r} tasks could not be sanitized for submit"
        raise TypeError(msg)
    if not isinstance(cleaned_clusters, list):
        msg = f"Job {inference_job_key!r} job_clusters could not be sanitized for submit"
        raise TypeError(msg)

    cleaned_tasks = render_job_parameter_refs(
        cleaned_tasks, parameter_values, run_id=db_run_id
    )
    cleaned_clusters = render_job_parameter_refs(
        cleaned_clusters, parameter_values, run_id=db_run_id
    )

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
        body["email_notifications"] = render_job_parameter_refs(
            email, parameter_values, run_id=db_run_id
        )

    run_as = parameter_overrides.get("ds_run_as", "").strip()
    if run_as:
        body["run_as"] = {"service_principal_name": run_as}

    acl = build_submit_access_control_list(parameter_overrides)
    if acl:
        body["access_control_list"] = acl

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
    monitor_url: str | None = None
    try:
        run = workspace_client.jobs.get_run(run_id=run_id)
        monitor_url = getattr(run, "run_page_url", None)
    except Exception as exc:
        logger.warning("Could not fetch run_page_url for run_id=%s: %s", run_id, exc)

    if monitor_url:
        logger.info(
            "Submitted versioned inference run_id=%s — monitor at %s",
            run_id,
            monitor_url,
        )
    else:
        cfg = workspace_client.config
        logger.info(
            "Submitted versioned inference run_id=%s — find in Workflows → search run id. "
            "(config host=%r is not always a clickable workspace URL)",
            run_id,
            getattr(cfg, "host", None),
        )
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
    wait_for_completion: bool = True,
    poll_interval_seconds: float = 30.0,
    wait_timeout_seconds: float | None = None,
    workspace_client: Any | None = None,
    logger: logging.Logger = LOGGER,
) -> int:
    """Load archived inference YAML from ``release_dir``, submit, and optionally wait."""
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
    if workspace_client is None and not dry_run:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    run_id = submit_inference_run(
        body,
        dry_run=dry_run,
        workspace_client=workspace_client,
        logger=logger,
    )
    if wait_for_completion and not dry_run and run_id:
        wait_for_inference_run(
            run_id,
            workspace_client=workspace_client,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=wait_timeout_seconds,
            logger=logger,
        )
    return run_id
