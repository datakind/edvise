"""
GenAI mapping pipeline — Databricks-facing helpers (run id resolution, stale runs, HITL timeout).

Entry jobs remain :mod:`edvise.genai.mapping.scripts.edvise_genai_ia` and
``edvise_genai_sma``; they call :func:`bootstrap_resolved_pipeline_run_id` at startup.
"""

from __future__ import annotations

import logging

from edvise.genai.mapping.state import pipeline_state
from edvise.genai.mapping.state.hitl_poller import (
    DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    HITLTimeoutError,
    poll_for_hitl_resolution,
)

LOGGER = logging.getLogger(__name__)


def bootstrap_resolved_pipeline_run_id(
    catalog: str,
    institution_id: str,
    pipeline_run_id_arg: str | None,
    *,
    stale_idle_minutes: int | None = None,
) -> str:
    """
    Reconcile stale ``running`` / ``awaiting_hitl`` rows, then resolve the active run id.

    Pass ``pipeline_run_id_arg`` as empty/None to use :func:`pipeline_state.resolve_pipeline_run_id`
    (implicit resume of ``running`` / ``awaiting_hitl`` / ``timed_out``, or a new suffixed id after
    ``complete`` / ``failed``).
    """
    idle = (
        stale_idle_minutes
        if stale_idle_minutes is not None
        else pipeline_state.STALE_PIPELINE_RUN_IDLE_MINUTES
    )
    pipeline_state.reconcile_stale_nonterminal_pipeline_runs(catalog, institution_id, idle)
    resolved = pipeline_state.resolve_pipeline_run_id(catalog, institution_id, pipeline_run_id_arg)
    LOGGER.info(
        "Resolved GenAI pipeline_run_id=%r (catalog=%r institution_id=%r stale_idle_minutes=%s)",
        resolved,
        catalog,
        institution_id,
        idle,
    )
    return resolved


def record_pipeline_run_timed_out_after_hitl_wait(
    catalog: str,
    institution_id: str,
    pipeline_run_id: str,
) -> None:
    """
    Best-effort ``timed_out`` on ``pipeline_runs`` when :class:`~edvise.genai.mapping.state.hitl_poller.HITLTimeoutError`
    propagates (``hitl_poller`` already writes ``timed_out``; this repeats the merge for durability).
    """
    try:
        pipeline_state.update_pipeline_run_status(catalog, institution_id, pipeline_run_id, "timed_out")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "Could not write timed_out to pipeline_runs after HITL timeout: catalog=%s run=%s (%s)",
            catalog,
            pipeline_run_id,
            e,
        )


def poll_uc_hitl_until_approved_or_timeout(
    catalog: str,
    institution_id: str,
    pipeline_run_id: str,
    phase: str,
    *,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """
    Wrap :func:`~edvise.genai.mapping.state.hitl_poller.poll_for_hitl_resolution`; on
    :class:`~edvise.genai.mapping.state.hitl_poller.HITLTimeoutError`, ensure ``timed_out`` on
    ``pipeline_runs`` then re-raise (``hitl_poller`` already updates UC state when ``institution_id`` is set).
    """
    try:
        return poll_for_hitl_resolution(
            catalog,
            pipeline_run_id,
            phase,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            institution_id=institution_id,
        )
    except HITLTimeoutError:
        record_pipeline_run_timed_out_after_hitl_wait(catalog, institution_id, pipeline_run_id)
        raise
