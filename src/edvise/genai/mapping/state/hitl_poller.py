"""
Poll Unity Catalog ``hitl_reviews`` until a phase is fully approved or fails fast on rejection/timeout.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from edvise.genai.mapping.state._sql import HITL_REVIEWS, get_spark_session, lit, qualified_table
from edvise.genai.mapping.state.pipeline_state import (
    check_hitl_resolution,
    log_phase_transition,
    update_pipeline_run_status,
)

LOGGER = logging.getLogger(__name__)

# Defaults for UC ``hitl_reviews`` polling; override via job parameters or kwargs.
DEFAULT_HITL_POLL_INTERVAL_SECONDS: int = 30
DEFAULT_HITL_POLL_TIMEOUT_SECONDS: int = 20 * 60


class HITLTimeoutError(Exception):
    """Raised when ``poll_for_hitl_resolution`` exceeds ``timeout_seconds`` without full approval."""


class HITLRejectedError(Exception):
    """Raised when any ``hitl_reviews`` row for the run and phase has ``status = rejected``."""


def _spark() -> Any:
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("No active Spark session; run on Databricks with Spark available")
    return spark


def _hitl_has_rejected(catalog: str, onboard_run_id: str, phase: str) -> bool:
    c = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    if not rid or not ph:
        raise ValueError("onboard_run_id and phase must be non-empty")

    t = qualified_table(c, HITL_REVIEWS)
    q = f"""
    SELECT COUNT(1) AS n_rejected
    FROM {t}
    WHERE onboard_run_id = {lit(rid)} AND phase = {lit(ph)} AND status = 'rejected'
    """
    n = int(_spark().sql(q).collect()[0]["n_rejected"])
    return n > 0


def poll_for_hitl_resolution(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
    *,
    institution_id: str | None = None,
) -> bool:
    """
    Block until ``check_hitl_resolution`` is True for the given run and phase.

    Sleeps ``poll_interval_seconds`` between checks. Logs each waiting iteration with elapsed time.
    Returns ``True`` when all rows for the run+phase are approved.

    If ``institution_id`` is set, a timeout first persists ``timed_out`` on ``pipeline_runs`` and
    ``pipeline_phases`` for this ``phase`` before raising.

    Raises:
        HITLRejectedError: If any matching row has ``status = rejected``.
        HITLTimeoutError: If ``timeout_seconds`` elapses without full approval.
    """
    c = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    if not rid or not ph:
        raise ValueError("onboard_run_id and phase must be non-empty")
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must be non-negative")
    if timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative")

    start = time.monotonic()
    deadline = start + float(timeout_seconds)

    while True:
        now = time.monotonic()
        elapsed = now - start

        if _hitl_has_rejected(c, rid, ph):
            raise HITLRejectedError(
                f"HITL rejected for catalog={c!r} onboard_run_id={rid!r} phase={ph!r} "
                f"(elapsed={elapsed:.1f}s)"
            )

        if check_hitl_resolution(c, rid, ph):
            return True

        if now >= deadline:
            inst = (institution_id or "").strip()
            if inst:
                try:
                    update_pipeline_run_status(c, inst, rid, "timed_out")
                    log_phase_transition(c, rid, ph, "timed_out")
                except Exception as e:  # noqa: BLE001 — best-effort before raising timeout
                    LOGGER.warning(
                        "Could not persist timed_out before HITL timeout: catalog=%s run=%s phase=%s (%s)",
                        c,
                        rid,
                        ph,
                        e,
                    )
            raise HITLTimeoutError(
                f"HITL not fully approved within {timeout_seconds}s for "
                f"catalog={c!r} onboard_run_id={rid!r} phase={ph!r} (elapsed={elapsed:.1f}s)"
            )

        LOGGER.info(
            "Waiting for HITL resolution: catalog=%s onboard_run_id=%s phase=%s elapsed=%.1fs",
            c,
            rid,
            ph,
            elapsed,
        )
        time.sleep(float(poll_interval_seconds))


def record_pipeline_run_timed_out_after_hitl_wait(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
) -> None:
    """
    Best-effort ``timed_out`` on ``pipeline_runs`` when :class:`HITLTimeoutError` propagates
    (``poll_for_hitl_resolution`` already writes ``timed_out`` when ``institution_id`` is set;
    this repeats the merge for durability).
    """
    try:
        update_pipeline_run_status(catalog, institution_id, onboard_run_id, "timed_out")
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "Could not write timed_out to pipeline_runs after HITL timeout: catalog=%s run=%s (%s)",
            catalog,
            onboard_run_id,
            e,
        )


def poll_uc_hitl_until_approved_or_timeout(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    phase: str,
    *,
    poll_interval_seconds: int = DEFAULT_HITL_POLL_INTERVAL_SECONDS,
    timeout_seconds: int = DEFAULT_HITL_POLL_TIMEOUT_SECONDS,
) -> bool:
    """
    Wrap :func:`poll_for_hitl_resolution`; on :class:`HITLTimeoutError`, ensure ``timed_out`` on
    ``pipeline_runs`` then re-raise.
    """
    try:
        return poll_for_hitl_resolution(
            catalog,
            onboard_run_id,
            phase,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            institution_id=institution_id,
        )
    except HITLTimeoutError:
        record_pipeline_run_timed_out_after_hitl_wait(catalog, institution_id, onboard_run_id)
        raise
