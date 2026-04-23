"""
Unity Catalog / Delta state for the GenAI mapping pipeline (infrastructure only).

State tables: ``{catalog}.genai_mapping.pipeline_runs`` (includes ``db_run_id`` for Databricks job
correlation), ``pipeline_phases``, ``hitl_reviews``.
All DML uses :meth:`pyspark.sql.SparkSession.sql` (no Delta Python API, no pandas).
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Any, Optional

from edvise.genai.mapping.state._sql import (
    HITL_REVIEWS,
    PIPELINE_PHASES,
    PIPELINE_RUNS,
    get_spark_session,
    lit,
    qualified_table,
)

LOGGER = logging.getLogger(__name__)

_INITIAL_RUN_STATUS: str = "running"

# Runs with no ``updated_at`` activity for this long while still ``running`` / ``awaiting_hitl``
# are treated as cluster/job timeouts (e.g. 2h Databricks limit) and marked ``timed_out``.
STALE_PIPELINE_RUN_IDLE_MINUTES: int = 25


def _spark() -> Any:
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("No active Spark session; run on Databricks with Spark available")
    return spark


def _sql_db_run_id(db_run_id: str | None) -> str:
    """SQL expression for ``db_run_id`` column (NULL or string literal)."""
    s = (db_run_id or "").strip()
    if not s:
        return "NULL"
    return lit(s)


def _row_to_plain_dict(row: Any) -> dict[str, Any]:
    """Coerce a Spark Row to JSON-friendly scalar dict (timestamps as ISO strings)."""
    d: dict[str, Any] = row.asDict() if hasattr(row, "asDict") else {}
    return {
        k: (v.isoformat() if isinstance(v, (datetime, date)) else v)
        for k, v in d.items()
    }


def create_pipeline_run(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    db_run_id: str | None = None,
) -> None:
    """
    Insert a new row into ``pipeline_runs`` with status ``running``.

    The table records which UC ``catalog`` the run belongs to. The operational convention
    is at most one *active* (non-terminal) run per ``institution_id``; that is not enforced
    in SQL and should be applied by the caller if needed.

    ``db_run_id`` stores the Databricks job run id (or other orchestration id) when provided.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    rid = str(onboard_run_id).strip()
    if not inst or not rid:
        raise ValueError("institution_id and onboard_run_id must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    db_sql = _sql_db_run_id(db_run_id)
    q = f"""
    INSERT INTO {t} (institution_id, onboard_run_id, `catalog`, status, db_run_id, created_at, updated_at)
    VALUES (
      {lit(inst)},
      {lit(rid)},
      {lit(c)},
      {lit(_INITIAL_RUN_STATUS)},
      {db_sql},
      current_timestamp(),
      current_timestamp()
    )
    """
    _spark().sql(q)


def update_pipeline_run_status(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    status: str,
    db_run_id: str | None = None,
) -> None:
    """
    Set ``status`` and ``updated_at`` for the run keyed by
    (``institution_id``, ``onboard_run_id``, ``catalog``).

    If no row matches, a new row is inserted (merge upsert) so a status update
    can follow a hand-created run id.

    On insert-only paths, ``db_run_id`` is set when non-empty; matched rows keep existing ``db_run_id``.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    rid = str(onboard_run_id).strip()
    st = str(status).strip()
    if not inst or not rid or not st:
        raise ValueError("institution_id, onboard_run_id, and status must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    db_sql = _sql_db_run_id(db_run_id)
    q = f"""
    MERGE INTO {t} t
    USING (
      SELECT
        {lit(inst)} AS institution_id,
        {lit(rid)} AS onboard_run_id,
        {lit(c)} AS ccat,
        {lit(st)} AS status,
        {db_sql} AS db_run_id
    ) s
    ON t.institution_id = s.institution_id
       AND t.onboard_run_id = s.onboard_run_id
       AND t.`catalog` = s.ccat
    WHEN MATCHED THEN
      UPDATE SET
        t.status = s.status,
        t.updated_at = current_timestamp()
    WHEN NOT MATCHED THEN
      INSERT (institution_id, onboard_run_id, `catalog`, status, db_run_id, created_at, updated_at)
      VALUES (s.institution_id, s.onboard_run_id, s.ccat, s.status, s.db_run_id, current_timestamp(), current_timestamp())
    """
    _spark().sql(q)


def get_latest_pipeline_run(
    catalog: str,
    institution_id: str,
) -> Optional[dict[str, Any]]:
    """
    Return the most recent ``pipeline_runs`` row for this institution in this catalog
    (by ``created_at`` descending), or ``None`` if none.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    q = f"""
    SELECT * FROM {t}
    WHERE institution_id = {lit(inst)} AND `catalog` = {lit(c)}
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = _spark().sql(q).collect()
    if not rows:
        return None
    return _row_to_plain_dict(rows[0])


def get_latest_pipeline_run_created_today(
    catalog: str,
    institution_id: str,
) -> Optional[dict[str, Any]]:
    """
    Most recent ``pipeline_runs`` row for this institution and catalog where
    ``to_date(created_at)`` equals ``current_date()``, or ``None``.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    q = f"""
    SELECT * FROM {t}
    WHERE institution_id = {lit(inst)}
      AND `catalog` = {lit(c)}
      AND to_date(created_at) = current_date()
    ORDER BY created_at DESC
    LIMIT 1
    """
    rows = _spark().sql(q).collect()
    if not rows:
        return None
    return _row_to_plain_dict(rows[0])


def count_pipeline_runs_created_today(catalog: str, institution_id: str) -> int:
    """Count ``pipeline_runs`` rows for this institution and catalog with ``created_at`` today."""
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    q = f"""
    SELECT COUNT(1) AS n FROM {t}
    WHERE institution_id = {lit(inst)}
      AND `catalog` = {lit(c)}
      AND to_date(created_at) = current_date()
    """
    n = int(_spark().sql(q).collect()[0]["n"])
    return n


def resolve_onboard_run_id(
    catalog: str,
    institution_id: str,
    onboard_run_id_override: Optional[str] = None,
) -> str:
    """
    Resolve the active ``onboard_run_id`` for this institution.

    If ``onboard_run_id_override`` is passed non-empty (manual override / explicit job id), it is
    returned unchanged. Otherwise the convention is ``{institution_id}_{YYYYMMDD}`` with optional
    numeric suffix for additional same-day runs after a terminal ``complete`` / ``failed`` state.
    """
    override = (onboard_run_id_override or "").strip()
    if override:
        return override

    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")

    base_id = f"{inst}_{date.today().strftime('%Y%m%d')}"
    latest = get_latest_pipeline_run_created_today(catalog, inst)
    if latest is None:
        return base_id

    st = str(latest.get("status") or "").strip()
    rid = str(latest.get("onboard_run_id") or "").strip()
    if not rid:
        return base_id

    if st in ("running", "awaiting_hitl", "timed_out"):
        return rid
    if st in ("complete", "failed"):
        n = count_pipeline_runs_created_today(catalog, inst)
        return f"{base_id}_{n + 1}"
    # Unknown legacy status: start a fresh suffixed id rather than reusing ambiguous rows.
    n = count_pipeline_runs_created_today(catalog, inst)
    return f"{base_id}_{n + 1}"


def new_execute_onboard_run_id(institution_id: str) -> str:
    """
    Fresh ``onboard_run_id`` for execute-mode jobs (distinct from onboard ids).

    Uses ``_exec_`` in the string so execute-mode bootstrap can resume the same id across
    multi-task jobs without colliding with onboard runs.
    """
    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    return f"{inst}_{date.today().strftime('%Y%m%d')}_exec_{uuid.uuid4().hex[:12]}"


def reconcile_stale_nonterminal_pipeline_runs(
    catalog: str,
    institution_id: str,
    idle_minutes: int,
) -> None:
    """
    Mark ``running`` / ``awaiting_hitl`` runs as ``timed_out`` when ``updated_at`` is older than
    ``idle_minutes`` (cluster preemption, Databricks job timeout, etc.).

    Matching ``pipeline_phases`` rows still in ``running`` or ``awaiting_hitl`` are set to
    ``timed_out`` with ``completed_at`` cleared.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    if idle_minutes < 1:
        raise ValueError("idle_minutes must be >= 1")

    pr = qualified_table(c, PIPELINE_RUNS)
    pp = qualified_table(c, PIPELINE_PHASES)
    threshold_sec = int(idle_minutes) * 60

    q_phases = f"""
    MERGE INTO {pp} AS t
    USING (
      SELECT DISTINCT onboard_run_id
      FROM {pr}
      WHERE institution_id = {lit(inst)}
        AND `catalog` = {lit(c)}
        AND status IN ('running', 'awaiting_hitl')
        AND updated_at < from_unixtime(unix_timestamp(current_timestamp()) - {threshold_sec})
    ) AS s
    ON t.onboard_run_id = s.onboard_run_id
       AND t.status IN ('running', 'awaiting_hitl')
    WHEN MATCHED THEN UPDATE SET
      t.status = 'timed_out',
      t.completed_at = NULL
    """
    _spark().sql(q_phases)

    q_runs = f"""
    UPDATE {pr}
    SET status = 'timed_out', updated_at = current_timestamp()
    WHERE institution_id = {lit(inst)}
      AND `catalog` = {lit(c)}
      AND status IN ('running', 'awaiting_hitl')
      AND updated_at < from_unixtime(unix_timestamp(current_timestamp()) - {threshold_sec})
    """
    _spark().sql(q_runs)


def log_phase_transition(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    status: str,
) -> None:
    """
    Insert or update a row in ``pipeline_phases`` keyed by (``onboard_run_id``, ``phase``).

    ``completed_at`` is set when ``status`` is one of: approved, rejected, complete; otherwise
    it is set to NULL (including ``timed_out``, which remains resumable). ``started_at`` is set on
    first insert and left unchanged on update.
    """
    c = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    st = str(status).strip()
    if not rid or not ph or not st:
        raise ValueError("onboard_run_id, phase, and status must be non-empty")

    t = qualified_table(c, PIPELINE_PHASES)
    q = f"""
    MERGE INTO {t} t
    USING (
      SELECT
        {lit(rid)} AS onboard_run_id,
        {lit(ph)} AS phase,
        {lit(st)} AS status
    ) s
    ON t.onboard_run_id = s.onboard_run_id AND t.phase = s.phase
    WHEN MATCHED THEN
      UPDATE SET
        t.status = s.status,
        t.completed_at = CASE
          WHEN s.status IN ('approved', 'rejected', 'complete') THEN current_timestamp()
          ELSE NULL
        END
    WHEN NOT MATCHED THEN
      INSERT (onboard_run_id, phase, status, started_at, completed_at)
      VALUES (
        s.onboard_run_id,
        s.phase,
        s.status,
        current_timestamp(),
        CASE
          WHEN s.status IN ('approved', 'rejected', 'complete') THEN current_timestamp()
          ELSE NULL
        END
      )
    """
    _spark().sql(q)


def register_hitl_artifacts(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifacts: list[dict[str, Any]],
) -> None:
    """
    For each item in ``artifacts`` (``artifact_type`` and ``artifact_path``), ensure a
    row exists in ``hitl_reviews`` with ``status = pending`` (merge by run, phase, type, path).
    """
    c = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    if not rid or not ph:
        raise ValueError("onboard_run_id and phase must be non-empty")
    if not artifacts:
        return

    rows: list[tuple[str, str]] = []
    for i, a in enumerate(artifacts):
        if not isinstance(a, dict):
            raise TypeError(f"artifacts[{i}] must be a dict, got {type(a).__name__!r}")
        at = a.get("artifact_type")
        ap = a.get("artifact_path")
        if at is None or ap is None:
            raise ValueError(f"artifacts[{i}] must have artifact_type and artifact_path")
        s_at = str(at).strip()
        s_ap = str(ap).strip()
        if not s_at or not s_ap:
            raise ValueError(f"artifacts[{i}]: artifact_type and artifact_path must be non-empty")
        rows.append((s_at, s_ap))

    t = qualified_table(c, HITL_REVIEWS)
    for s_at, s_ap in rows:
        q = f"""
        MERGE INTO {t} t
        USING (
          SELECT
            {lit(rid)} AS onboard_run_id,
            {lit(ph)} AS phase,
            {lit(s_at)} AS artifact_type,
            {lit(s_ap)} AS artifact_path
        ) s
        ON t.onboard_run_id = s.onboard_run_id
           AND t.phase = s.phase
           AND t.artifact_type = s.artifact_type
           AND t.artifact_path = s.artifact_path
        WHEN MATCHED THEN
          UPDATE SET
            t.status = 'pending',
            t.reviewer = NULL,
            t.reviewed_at = NULL
        WHEN NOT MATCHED THEN
          INSERT (onboard_run_id, phase, artifact_type, artifact_path, status, reviewer, reviewed_at)
          VALUES (s.onboard_run_id, s.phase, s.artifact_type, s.artifact_path, 'pending', NULL, NULL)
        """
        _spark().sql(q)


def check_hitl_resolution(
    catalog: str,
    onboard_run_id: str,
    phase: str,
) -> bool:
    """
    Return True iff there is at least one ``hitl_reviews`` row for the run+phase
    and every such row has ``status = approved``; False if there are no rows or if any
    is not approved.
    """
    c = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    if not rid or not ph:
        raise ValueError("onboard_run_id and phase must be non-empty")

    t = qualified_table(c, HITL_REVIEWS)
    q = f"""
    SELECT
      COUNT(1) AS n_total,
      COALESCE(
        SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END),
        0
      ) AS n_approved
    FROM {t}
    WHERE onboard_run_id = {lit(rid)} AND phase = {lit(ph)}
    """
    d = _spark().sql(q).collect()[0].asDict()
    n_total = int(d["n_total"])
    n_approved = int(d["n_approved"])
    if n_total == 0:
        return False
    return n_approved == n_total


def resolve_hitl(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    reviewer: str,
    status: str,
) -> None:
    """
    Set ``status``, ``reviewer``, and ``reviewed_at`` on all ``hitl_reviews`` rows
    for the given run, phase, and ``artifact_type`` (e.g. multiple files of the same type
    with different ``artifact_path`` values are updated together).
    """
    c = str(catalog).strip()
    rid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    at = str(artifact_type).strip()
    st = str(status).strip()
    rev = str(reviewer).strip() if reviewer is not None else None
    if not rid or not ph or not at or not st:
        raise ValueError("onboard_run_id, phase, artifact_type, and status must be non-empty")
    if st not in ("approved", "rejected"):
        raise ValueError("status must be 'approved' or 'rejected'")

    t = qualified_table(c, HITL_REVIEWS)
    if rev:
        reviewer_set = f"reviewer = {lit(rev)}"
    else:
        reviewer_set = "reviewer = NULL"

    q = f"""
    UPDATE {t}
    SET
      status = {lit(st)},
      {reviewer_set},
      reviewed_at = current_timestamp()
    WHERE onboard_run_id = {lit(rid)}
      AND phase = {lit(ph)}
      AND artifact_type = {lit(at)}
    """
    _spark().sql(q)


def bootstrap_resolved_onboard_run_id(
    catalog: str,
    institution_id: str,
    onboard_run_id_arg: str | None,
    *,
    stale_idle_minutes: int | None = None,
) -> str:
    """
    Reconcile stale ``running`` / ``awaiting_hitl`` rows, then resolve the active run id.

    Pass ``onboard_run_id_arg`` as empty/None to use :func:`resolve_onboard_run_id`
    (implicit resume of ``running`` / ``awaiting_hitl`` / ``timed_out``, or a new suffixed id after
    ``complete`` / ``failed``).
    """
    idle = (
        stale_idle_minutes
        if stale_idle_minutes is not None
        else STALE_PIPELINE_RUN_IDLE_MINUTES
    )
    reconcile_stale_nonterminal_pipeline_runs(catalog, institution_id, idle)
    resolved = resolve_onboard_run_id(catalog, institution_id, onboard_run_id_arg)
    LOGGER.info(
        "Resolved GenAI onboard_run_id=%r (catalog=%r institution_id=%r stale_idle_minutes=%s)",
        resolved,
        catalog,
        institution_id,
        idle,
    )
    return resolved


def bootstrap_resolved_onboard_run_id_for_execute(
    catalog: str,
    institution_id: str,
    *,
    db_run_id: str | None = None,
    stale_idle_minutes: int | None = None,
) -> str:
    """
    Reconcile stale rows, then return an execute ``onboard_run_id``.

    If today's latest run for this institution is a non-terminal ``*_exec_*`` row (same multi-task
    job: IA then SMA), that id is reused. Otherwise a new id is minted and inserted into
    ``pipeline_runs`` so downstream tasks can resolve it the same way onboard does.
    """
    idle = (
        stale_idle_minutes
        if stale_idle_minutes is not None
        else STALE_PIPELINE_RUN_IDLE_MINUTES
    )
    reconcile_stale_nonterminal_pipeline_runs(catalog, institution_id, idle)
    latest = get_latest_pipeline_run_created_today(catalog, institution_id)
    if latest:
        st = str(latest.get("status") or "").strip()
        rid = str(latest.get("onboard_run_id") or "").strip()
        if rid and "_exec_" in rid and st in ("running", "awaiting_hitl", "timed_out"):
            LOGGER.info(
                "Resuming GenAI execute onboard_run_id=%r (catalog=%r institution_id=%r)",
                rid,
                catalog,
                institution_id,
            )
            return rid

    rid = new_execute_onboard_run_id(institution_id)
    create_pipeline_run(catalog, institution_id, rid, db_run_id=db_run_id)
    LOGGER.info(
        "Resolved GenAI execute onboard_run_id=%r (catalog=%r institution_id=%r stale_idle_minutes=%s)",
        rid,
        catalog,
        institution_id,
        idle,
    )
    return rid


def mark_execute_pipeline_run_status(
    catalog: str,
    institution_id: str,
    onboard_run_id: str,
    status: str,
) -> None:
    """Best-effort ``pipeline_runs`` status for execute jobs (failure / ignored errors)."""
    try:
        update_pipeline_run_status(catalog, institution_id, onboard_run_id, status)
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "Could not write execute pipeline_runs status=%s catalog=%s run=%s (%s)",
            status,
            catalog,
            onboard_run_id,
            e,
        )
