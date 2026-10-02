"""ES dual-pin / segmented ``runs/submit`` orchestration (PDP submit path unchanged)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edvise.runtime.versioned_inference.bundle.from_dab import (
    inference_yml_path,
    load_inference_job_definition,
)
from edvise.runtime.versioned_inference.child_run_values import (
    DRY_RUN_INGESTION_HANDOFF,
    fetch_ingestion_handoff_from_run,
    reconstruct_ingestion_handoff,
)
from edvise.runtime.versioned_inference.dab_layout import (
    genai_execute_dab_bundle_layout,
    genai_snapshot_dir,
    resolve_dab_bundle_layout,
)
from edvise.runtime.versioned_inference.es_segments import (
    DATA_INGESTION_TASK_KEY,
    INGESTION_HANDOFF_KEYS,
    apply_ingestion_handoff_to_job,
    es_full_task_keys,
    es_inference_task_keys,
    es_ingestion_task_keys,
    job_with_selected_tasks,
)
from edvise.runtime.versioned_inference.genai_registry import (
    resolve_genai_pipeline_version_from_registry,
)
from edvise.runtime.versioned_inference.io_chain import (
    assert_es_inference_outputs_ready,
    assert_genai_execute_outputs_ready,
    assert_ingestion_outputs_ready,
)
from edvise.runtime.versioned_inference.parameters import (
    resolve_versioned_job_parameters,
)
from edvise.runtime.versioned_inference.pipeline_version_ref import git_ref_kind
from edvise.runtime.versioned_inference.submit import (
    DEFAULT_GIT_URL,
    build_submit_run_body,
    fetch_run_page_url,
    log_child_run_monitor_url,
    submit_inference_run,
    wait_for_inference_run,
)

LOGGER = logging.getLogger(__name__)

_ES_SCHEMA = "edvise"


@dataclass(frozen=True)
class EsChildRunIds:
    """Child run ids produced by the ES versioned trigger."""

    is_genai: bool
    # Non-GenAI: one full ES spark DAG.
    es_full: int | None = None
    # GenAI dual-pin segments.
    es_ingestion: int | None = None
    genai_execute: int | None = None
    es_inference: int | None = None
    # Optional monitor URLs (logged alone so Databricks keeps them clickable).
    es_full_url: str | None = None
    es_ingestion_url: str | None = None
    genai_execute_url: str | None = None
    es_inference_url: str | None = None

    @property
    def primary(self) -> int:
        """Id used for launcher ``child_inference_run_id`` (final ES segment when split)."""
        for value in (
            self.es_inference,
            self.es_full,
            self.es_ingestion,
            self.genai_execute,
        ):
            if value is not None:
                return int(value)
        return 0

    def as_payload(self) -> dict[str, str]:
        """
        Structured child-run ids for launcher events / logs.

        Non-GenAI: ``child_run_es_full``
        GenAI: ``child_run_es_ingestion``, ``child_run_genai_execute``,
        ``child_run_es_inference``
        Always: ``child_inference_run_id`` (final ES run).
        """
        out: dict[str, str] = {
            "is_genai_institution": "true" if self.is_genai else "false"
        }
        if self.is_genai:
            if self.es_ingestion is not None:
                out["child_run_es_ingestion"] = str(self.es_ingestion)
            if self.genai_execute is not None:
                out["child_run_genai_execute"] = str(self.genai_execute)
            if self.es_inference is not None:
                out["child_run_es_inference"] = str(self.es_inference)
        else:
            if self.es_full is not None:
                out["child_run_es_full"] = str(self.es_full)
        out["child_inference_run_id"] = str(self.primary)
        return out

    def log_monitor_urls(self, logger: logging.Logger = LOGGER) -> None:
        """Re-emit each child URL as a bare stdout line (clickable in Jobs UI)."""
        from edvise.runtime.versioned_inference.submit import log_child_run_monitor_url

        pairs = (
            ("es_full", self.es_full, self.es_full_url),
            ("es_ingestion", self.es_ingestion, self.es_ingestion_url),
            ("genai_execute", self.genai_execute, self.genai_execute_url),
            ("es_inference", self.es_inference, self.es_inference_url),
        )
        for label, run_id, url in pairs:
            if run_id is None:
                continue
            logger.info("Child run summary (%s)", label)
            log_child_run_monitor_url(run_id, url, logger=logger, status="summary")


@dataclass
class EsSubmitPlan:
    """Resolved parameters and bodies for ES-full or dual-pin GenAI paths."""

    is_genai: bool
    es_pipeline_version: str
    genai_pipeline_version: str | None
    es_parameters: dict[str, str]
    es_full_body: dict[str, Any] | None = None
    ingestion_body: dict[str, Any] | None = None
    genai_body: dict[str, Any] | None = None
    inference_body: dict[str, Any] | None = None
    handoff: dict[str, str] = field(default_factory=dict)


def _resolve_es_job_parameters(
    job: dict[str, Any],
    release_dir: Path,
    *,
    parameter_overrides: dict[str, str],
    extra_parameter_overrides: dict[str, str] | None,
    stable_trigger: dict[str, Any] | None,
    es_pipeline_version: str,
    logger: logging.Logger,
) -> dict[str, str]:
    overrides = dict(parameter_overrides)
    overrides.setdefault("schema_type", _ES_SCHEMA)
    overrides["pipeline_version"] = es_pipeline_version
    return resolve_versioned_job_parameters(
        job,
        release_dir,
        launcher_overrides=overrides,
        extra_overrides=extra_parameter_overrides,
        stable_trigger=stable_trigger,
        logger=logger,
    )


def build_genai_execute_parameter_overrides(
    es_parameters: dict[str, str],
    *,
    genai_pipeline_version: str,
    bronze_batch_dir: str,
    access_control_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map ES launcher params → GenAI execute job parameter names (original contract)."""
    db_run = str(es_parameters.get("db_run_id") or "").strip()
    overrides: dict[str, str] = {
        "institution_id": es_parameters.get("databricks_institution_name", ""),
        "catalog": es_parameters.get("DB_workspace", ""),
        "inputs_toml_path": es_parameters.get("genai_inputs_toml_path", "")
        or "inputs.toml",
        "pipeline_version": genai_pipeline_version,
        # Match github_es_inference.yml nested job_parameters.db_run_id suffix.
        "db_run_id": f"{db_run}_genai_execute" if db_run else "",
        "bronze_batch_dir": bronze_batch_dir,
    }
    acl_src = (
        access_control_overrides
        if access_control_overrides is not None
        else es_parameters
    )
    for key in ("ds_run_as", "datakind_group_to_manage_workflow", "viewer_user"):
        val = str(acl_src.get(key, "") or "").strip()
        if val:
            overrides[key] = val
    return overrides


def _run_name(inst: str, model: str, label: str, version: str) -> str:
    return f"versioned-es-{label}-{inst}-{model}-{version[:12]}"


def build_segment_submit_body(
    job: dict[str, Any],
    *,
    keep_keys: set[str],
    pipeline_version: str,
    git_url: str,
    run_name: str,
    parameter_overrides: dict[str, str],
    access_control_overrides: dict[str, str] | None = None,
    inference_job_key: str,
    handoff: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Filter tasks, optionally rewrite ingestion task-value refs, build submit body."""
    segmented = job_with_selected_tasks(job, keep_keys)
    if handoff:
        segmented = apply_ingestion_handoff_to_job(segmented, handoff)
    return build_submit_run_body(
        segmented,
        pipeline_version=pipeline_version,
        git_url=git_url,
        run_name=run_name,
        parameter_overrides=parameter_overrides,
        access_control_overrides=access_control_overrides,
        inference_job_key=inference_job_key,
    )


def plan_es_versioned_submit(
    release_dir: Path,
    *,
    es_pipeline_version: str,
    is_genai: bool,
    parameter_overrides: dict[str, str],
    extra_parameter_overrides: dict[str, str] | None = None,
    stable_trigger: dict[str, Any] | None = None,
    git_url: str = DEFAULT_GIT_URL,
    genai_pipeline_version: str | None = None,
    handoff: dict[str, str] | None = None,
    logger: logging.Logger = LOGGER,
) -> EsSubmitPlan:
    """
    Build ES-full or dual-pin submit bodies from archived snapshots.

    For GenAI, ``handoff`` may be omitted when only planning the ingestion body;
    ``genai_body`` / ``inference_body`` require handoff (or dry-run placeholders).
    """
    es_layout = resolve_dab_bundle_layout(_ES_SCHEMA)
    es_yml = inference_yml_path(release_dir, es_layout.inference_yml_snapshot_rel)
    es_job = load_inference_job_definition(es_yml, job_key=es_layout.inference_job_key)
    es_params = _resolve_es_job_parameters(
        es_job,
        release_dir,
        parameter_overrides=parameter_overrides,
        extra_parameter_overrides=extra_parameter_overrides,
        stable_trigger=stable_trigger,
        es_pipeline_version=es_pipeline_version,
        logger=logger,
    )
    inst = es_params.get("databricks_institution_name", "unknown")
    model = es_params.get("model_name", "unknown")
    acl = parameter_overrides

    plan = EsSubmitPlan(
        is_genai=is_genai,
        es_pipeline_version=es_pipeline_version,
        genai_pipeline_version=genai_pipeline_version,
        es_parameters=es_params,
    )

    if not is_genai:
        plan.es_full_body = build_segment_submit_body(
            es_job,
            keep_keys=es_full_task_keys(es_job.get("tasks") or []),
            pipeline_version=es_pipeline_version,
            git_url=git_url,
            run_name=_run_name(inst, model, "full", es_pipeline_version),
            parameter_overrides=es_params,
            access_control_overrides=acl,
            inference_job_key=es_layout.inference_job_key,
        )
        return plan

    if not genai_pipeline_version:
        raise ValueError(
            "genai_pipeline_version is required when is_genai_institution is true"
        )

    plan.ingestion_body = build_segment_submit_body(
        es_job,
        keep_keys=es_ingestion_task_keys(),
        pipeline_version=es_pipeline_version,
        git_url=git_url,
        run_name=_run_name(inst, model, "ingestion", es_pipeline_version),
        parameter_overrides=es_params,
        access_control_overrides=acl,
        inference_job_key=es_layout.inference_job_key,
    )

    effective_handoff = handoff
    if effective_handoff is None:
        return plan

    plan.handoff = dict(effective_handoff)
    genai_layout = genai_execute_dab_bundle_layout()
    genai_dir = genai_snapshot_dir(release_dir)
    genai_yml = inference_yml_path(genai_dir, genai_layout.inference_yml_snapshot_rel)
    genai_job = load_inference_job_definition(
        genai_yml, job_key=genai_layout.inference_job_key
    )
    genai_overrides = build_genai_execute_parameter_overrides(
        es_params,
        genai_pipeline_version=genai_pipeline_version,
        bronze_batch_dir=str(effective_handoff.get("bronze_batch_dir", "")),
        access_control_overrides=acl,
    )
    genai_params = resolve_versioned_job_parameters(
        genai_job,
        genai_dir,
        launcher_overrides=genai_overrides,
        extra_overrides=None,
        stable_trigger=None,
        logger=logger,
    )
    genai_keys = {
        str(t.get("task_key", "")).strip()
        for t in (genai_job.get("tasks") or [])
        if isinstance(t, dict) and str(t.get("task_key", "")).strip()
    }
    plan.genai_body = build_segment_submit_body(
        genai_job,
        keep_keys=genai_keys,
        pipeline_version=genai_pipeline_version,
        git_url=git_url,
        run_name=_run_name(inst, model, "genai", genai_pipeline_version),
        parameter_overrides=genai_params,
        access_control_overrides=acl,
        inference_job_key=genai_layout.inference_job_key,
    )

    plan.inference_body = build_segment_submit_body(
        es_job,
        keep_keys=es_inference_task_keys(es_job.get("tasks") or []),
        pipeline_version=es_pipeline_version,
        git_url=git_url,
        run_name=_run_name(inst, model, "inference", es_pipeline_version),
        parameter_overrides=es_params,
        access_control_overrides=acl,
        inference_job_key=es_layout.inference_job_key,
        handoff=effective_handoff,
    )
    return plan


def _submit_and_maybe_wait(
    body: dict[str, Any],
    *,
    dry_run: bool,
    wait: bool,
    poll_interval_seconds: float,
    wait_timeout_seconds: float | None,
    workspace_client: Any | None,
    logger: logging.Logger,
) -> tuple[int, str | None]:
    run_id = submit_inference_run(
        body,
        dry_run=dry_run,
        workspace_client=workspace_client,
        logger=logger,
    )
    url: str | None = None
    if not dry_run and run_id and workspace_client is not None:
        url = fetch_run_page_url(workspace_client, run_id, logger=logger)
    if wait and not dry_run and run_id:
        wait_for_inference_run(
            run_id,
            workspace_client=workspace_client,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=wait_timeout_seconds,
            logger=logger,
        )
        # Re-emit URL after long polling so it stays findable / clickable.
        if url:
            log_child_run_monitor_url(run_id, url, logger=logger, status="succeeded")
        elif workspace_client is not None:
            url = fetch_run_page_url(workspace_client, run_id, logger=logger)
            log_child_run_monitor_url(run_id, url, logger=logger, status="succeeded")
    return run_id, url


def resolve_handoff_after_ingestion(
    *,
    dry_run: bool,
    ingestion_run_id: int,
    es_parameters: dict[str, str],
    workspace_client: Any | None,
    logger: logging.Logger = LOGGER,
) -> dict[str, str]:
    """Jobs API values when available; otherwise reconstruct using original path helpers."""
    if dry_run:
        logger.info("dry-run: using placeholder ingestion handoff")
        return dict(DRY_RUN_INGESTION_HANDOFF)

    reconstructed = reconstruct_ingestion_handoff(
        db_workspace=es_parameters.get("DB_workspace", ""),
        databricks_institution_name=es_parameters.get(
            "databricks_institution_name", ""
        ),
        model_name=es_parameters.get("model_name", ""),
        batch_id=es_parameters.get("batch_id", ""),
        logger=logger,
    )
    if workspace_client is None or not ingestion_run_id:
        return require_keys(reconstructed)

    try:
        from_api = fetch_ingestion_handoff_from_run(
            workspace_client,
            ingestion_run_id,
            task_key=DATA_INGESTION_TASK_KEY,
            required_keys=INGESTION_HANDOFF_KEYS,
            hard_required=("config_file_path",),
            logger=logger,
        )
    except Exception as exc:
        logger.warning(
            "Could not read data_ingestion task values from run_id=%s (%s); "
            "using reconstructed handoff.",
            ingestion_run_id,
            exc,
        )
        return require_keys(reconstructed)

    merged = dict(reconstructed)
    for key, val in from_api.items():
        if str(val or "").strip():
            merged[key] = str(val)
    return require_keys(merged)


def require_keys(handoff: dict[str, str]) -> dict[str, str]:
    from edvise.runtime.versioned_inference.child_run_values import (
        require_ingestion_handoff,
    )

    return require_ingestion_handoff(
        handoff,
        required_keys=INGESTION_HANDOFF_KEYS,
        hard_required=("config_file_path",),
    )


def submit_es_versioned_inference_from_bundle(
    release_dir: Path,
    *,
    es_pipeline_version: str,
    is_genai: bool,
    parameter_overrides: dict[str, str],
    extra_parameter_overrides: dict[str, str] | None = None,
    stable_trigger: dict[str, Any] | None = None,
    git_url: str = DEFAULT_GIT_URL,
    db_workspace: str = "",
    databricks_institution_name: str = "",
    model_run_id: str = "",
    dry_run: bool = False,
    wait_for_completion: bool = True,
    poll_interval_seconds: float = 30.0,
    wait_timeout_seconds: float | None = None,
    workspace_client: Any | None = None,
    logger: logging.Logger = LOGGER,
) -> EsChildRunIds:
    """
    Non-GenAI: one spark-only ES-full child run @ ``es_pipeline_version``.

    GenAI: ES ingestion → GenAI execute @ registry → ES inference (shared ``db_run_id``).
    Hard I/O checks link bronze → GenAI pipeline_input → silver inference.
    """
    db_ws = db_workspace or parameter_overrides.get("DB_workspace", "")
    inst = databricks_institution_name or parameter_overrides.get(
        "databricks_institution_name", ""
    )
    db_run_id = str(parameter_overrides.get("db_run_id", "") or "").strip()

    genai_version: str | None = None
    if is_genai:
        genai_version = resolve_genai_pipeline_version_from_registry(
            db_ws,
            inst,
            logger=logger,
        )
        genai_dir = genai_snapshot_dir(release_dir)
        if not genai_dir.is_dir():
            raise FileNotFoundError(
                f"GenAI snapshot missing under {genai_dir}; run materialize first."
            )

    if workspace_client is None and not dry_run:
        from databricks.sdk import WorkspaceClient

        workspace_client = WorkspaceClient()

    if not is_genai:
        plan = plan_es_versioned_submit(
            release_dir,
            es_pipeline_version=es_pipeline_version,
            is_genai=False,
            parameter_overrides=parameter_overrides,
            extra_parameter_overrides=extra_parameter_overrides,
            stable_trigger=stable_trigger,
            git_url=git_url,
            logger=logger,
        )
        assert plan.es_full_body is not None
        logger.info(
            "ES-full submit at git %s %s (%s tasks)",
            git_ref_kind(es_pipeline_version),
            es_pipeline_version,
            len(plan.es_full_body.get("tasks") or []),
        )
        run_id, url = _submit_and_maybe_wait(
            plan.es_full_body,
            dry_run=dry_run,
            wait=wait_for_completion,
            poll_interval_seconds=poll_interval_seconds,
            wait_timeout_seconds=wait_timeout_seconds,
            workspace_client=workspace_client,
            logger=logger,
        )
        if not dry_run and wait_for_completion and model_run_id.strip():
            assert_es_inference_outputs_ready(
                db_ws,
                inst,
                model_run_id,
                db_run_id=db_run_id,
                logger=logger,
            )
        child = EsChildRunIds(is_genai=False, es_full=run_id, es_full_url=url)
        child.log_monitor_urls(logger)
        return child

    # --- GenAI dual-pin ---
    ingestion_plan = plan_es_versioned_submit(
        release_dir,
        es_pipeline_version=es_pipeline_version,
        is_genai=True,
        parameter_overrides=parameter_overrides,
        extra_parameter_overrides=extra_parameter_overrides,
        stable_trigger=stable_trigger,
        git_url=git_url,
        genai_pipeline_version=genai_version,
        handoff=None,
        logger=logger,
    )
    assert ingestion_plan.ingestion_body is not None
    logger.info(
        "ES ingestion (data_ingestion) at git %s %s",
        git_ref_kind(es_pipeline_version),
        es_pipeline_version,
    )
    ingestion_id, ingestion_url = _submit_and_maybe_wait(
        ingestion_plan.ingestion_body,
        dry_run=dry_run,
        wait=True,
        poll_interval_seconds=poll_interval_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        workspace_client=workspace_client,
        logger=logger,
    )

    handoff = resolve_handoff_after_ingestion(
        dry_run=dry_run,
        ingestion_run_id=ingestion_id,
        es_parameters=ingestion_plan.es_parameters,
        workspace_client=workspace_client,
        logger=logger,
    )
    if not dry_run:
        assert_ingestion_outputs_ready(handoff, logger=logger)

    plan = plan_es_versioned_submit(
        release_dir,
        es_pipeline_version=es_pipeline_version,
        is_genai=True,
        parameter_overrides=parameter_overrides,
        extra_parameter_overrides=extra_parameter_overrides,
        stable_trigger=stable_trigger,
        git_url=git_url,
        genai_pipeline_version=genai_version,
        handoff=handoff,
        logger=logger,
    )
    assert plan.genai_body is not None and plan.inference_body is not None

    logger.info(
        "GenAI execute at git %s %s (bronze_batch_dir=%r)",
        git_ref_kind(genai_version or ""),
        genai_version,
        handoff.get("bronze_batch_dir", ""),
    )
    genai_id, genai_url = _submit_and_maybe_wait(
        plan.genai_body,
        dry_run=dry_run,
        wait=True,
        poll_interval_seconds=poll_interval_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        workspace_client=workspace_client,
        logger=logger,
    )
    if not dry_run:
        assert_genai_execute_outputs_ready(
            db_ws,
            inst,
            bronze_batch_dir=handoff.get("bronze_batch_dir", ""),
            logger=logger,
        )

    logger.info(
        "ES inference (data_audit…output_publish) at git %s %s",
        git_ref_kind(es_pipeline_version),
        es_pipeline_version,
    )
    inference_id, inference_url = _submit_and_maybe_wait(
        plan.inference_body,
        dry_run=dry_run,
        wait=wait_for_completion,
        poll_interval_seconds=poll_interval_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        workspace_client=workspace_client,
        logger=logger,
    )
    if not dry_run and wait_for_completion and model_run_id.strip():
        assert_es_inference_outputs_ready(
            db_ws,
            inst,
            model_run_id,
            db_run_id=db_run_id,
            logger=logger,
        )

    child = EsChildRunIds(
        is_genai=True,
        es_ingestion=ingestion_id,
        genai_execute=genai_id,
        es_inference=inference_id,
        es_ingestion_url=ingestion_url,
        genai_execute_url=genai_url,
        es_inference_url=inference_url,
    )
    child.log_monitor_urls(logger)
    return child
