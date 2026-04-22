"""
Unity Catalog / Delta state for the GenAI mapping pipeline (infrastructure only).

State tables: ``{catalog}.genai_mapping.pipeline_runs|pipeline_phases|hitl_reviews``.
All DML uses :meth:`pyspark.sql.SparkSession.sql` (no Delta Python API, no pandas).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Mapping, Optional

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
_TERMINAL_PHASE_STATUSES: frozenset[str] = frozenset(
    ("approved", "rejected", "complete"),
)


def _spark() -> Any:
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("No active Spark session; run on Databricks with Spark available")
    return spark


def _row_to_plain_dict(row: Any) -> dict[str, Any]:
    """Coerce a Spark Row / Row-like to JSON-friendly dicts."""
    if row is None:
        return {}
    d = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)  # type: ignore[call-overload]
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (datetime, date)):
            out[k] = v.isoformat() if hasattr(v, "isoformat") else v
        elif isinstance(v, dict):
            out[k] = {ik: (iv.isoformat() if isinstance(iv, (datetime, date)) else iv) for ik, iv in v.items()}  # type: ignore[union-attr]
        else:
            out[k] = v
    return out


def create_pipeline_run(
    catalog: str,
    institution_id: str,
    pipeline_run_id: str,
) -> None:
    """
    Insert a new row into ``pipeline_runs`` with status ``running``.

    The table records which UC ``catalog`` the run belongs to. The operational convention
    is at most one *active* (non-terminal) run per ``institution_id``; that is not enforced
    in SQL and should be applied by the caller if needed.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    rid = str(pipeline_run_id).strip()
    if not inst or not rid:
        raise ValueError("institution_id and pipeline_run_id must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    q = f"""
    INSERT INTO {t} (institution_id, pipeline_run_id, `catalog`, status, created_at, updated_at)
    VALUES (
      {lit(inst)},
      {lit(rid)},
      {lit(c)},
      {lit(_INITIAL_RUN_STATUS)},
      current_timestamp(),
      current_timestamp()
    )
    """
    _spark().sql(q)


def update_pipeline_run_status(
    catalog: str,
    institution_id: str,
    pipeline_run_id: str,
    status: str,
) -> None:
    """
    Set ``status`` and ``updated_at`` for the run keyed by
    (``institution_id``, ``pipeline_run_id``, ``catalog``).

    If no row matches, a new row is inserted (merge upsert) so a status update
    can follow a hand-created run id.
    """
    c = str(catalog).strip()
    inst = str(institution_id).strip()
    rid = str(pipeline_run_id).strip()
    st = str(status).strip()
    if not inst or not rid or not st:
        raise ValueError("institution_id, pipeline_run_id, and status must be non-empty")

    t = qualified_table(c, PIPELINE_RUNS)
    q = f"""
    MERGE INTO {t} t
    USING (
      SELECT
        {lit(inst)} AS institution_id,
        {lit(rid)} AS pipeline_run_id,
        {lit(c)} AS ccat,
        {lit(st)} AS status
    ) s
    ON t.institution_id = s.institution_id
       AND t.pipeline_run_id = s.pipeline_run_id
       AND t.`catalog` = s.ccat
    WHEN MATCHED THEN
      UPDATE SET
        t.status = s.status,
        t.updated_at = current_timestamp()
    WHEN NOT MATCHED THEN
      INSERT (institution_id, pipeline_run_id, `catalog`, status, created_at, updated_at)
      VALUES (s.institution_id, s.pipeline_run_id, s.ccat, s.status, current_timestamp(), current_timestamp())
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


def log_phase_transition(
    catalog: str,
    pipeline_run_id: str,
    phase: str,
    status: str,
) -> None:
    """
    Insert or update a row in ``pipeline_phases`` keyed by (``pipeline_run_id``, ``phase``).

    ``completed_at`` is set when ``status`` is one of: approved, rejected, complete; otherwise
    it is set to NULL. ``started_at`` is set on first insert and left unchanged on update.
    """
    c = str(catalog).strip()
    rid = str(pipeline_run_id).strip()
    ph = str(phase).strip()
    st = str(status).strip()
    if not rid or not ph or not st:
        raise ValueError("pipeline_run_id, phase, and status must be non-empty")

    t = qualified_table(c, PIPELINE_PHASES)
    q = f"""
    MERGE INTO {t} t
    USING (
      SELECT
        {lit(rid)} AS pipeline_run_id,
        {lit(ph)} AS phase,
        {lit(st)} AS status
    ) s
    ON t.pipeline_run_id = s.pipeline_run_id AND t.phase = s.phase
    WHEN MATCHED THEN
      UPDATE SET
        t.status = s.status,
        t.completed_at = CASE
          WHEN s.status IN ('approved', 'rejected', 'complete') THEN current_timestamp()
          ELSE NULL
        END
    WHEN NOT MATCHED THEN
      INSERT (pipeline_run_id, phase, status, started_at, completed_at)
      VALUES (
        s.pipeline_run_id,
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
    pipeline_run_id: str,
    phase: str,
    artifacts: list[dict],
) -> None:
    """
    For each item in ``artifacts`` (``artifact_type`` and ``artifact_path``), ensure a
    row exists in ``hitl_reviews`` with ``status = pending`` (merge by run, phase, type, path).
    """
    c = str(catalog).strip()
    rid = str(pipeline_run_id).strip()
    ph = str(phase).strip()
    if not rid or not ph:
        raise ValueError("pipeline_run_id and phase must be non-empty")
    if not artifacts:
        return

    rows: list[tuple[str, str]] = []
    for i, a in enumerate(artifacts):
        if not isinstance(a, Mapping):
            raise TypeError(f"artifacts[{i}] must be a mapping, got {type(a).__name__!r}")
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
            {lit(rid)} AS pipeline_run_id,
            {lit(ph)} AS phase,
            {lit(s_at)} AS artifact_type,
            {lit(s_ap)} AS artifact_path
        ) s
        ON t.pipeline_run_id = s.pipeline_run_id
           AND t.phase = s.phase
           AND t.artifact_type = s.artifact_type
           AND t.artifact_path = s.artifact_path
        WHEN MATCHED THEN
          UPDATE SET
            t.status = 'pending',
            t.reviewer = NULL,
            t.reviewed_at = NULL
        WHEN NOT MATCHED THEN
          INSERT (pipeline_run_id, phase, artifact_type, artifact_path, status, reviewer, reviewed_at)
          VALUES (s.pipeline_run_id, s.phase, s.artifact_type, s.artifact_path, 'pending', NULL, NULL)
        """
        _spark().sql(q)


def check_hitl_resolution(
    catalog: str,
    pipeline_run_id: str,
    phase: str,
) -> bool:
    """
    Return True iff there is at least one ``hitl_reviews`` row for the run+phase
    and every such row has ``status = approved``; False if there are no rows or if any
    is not approved.
    """
    c = str(catalog).strip()
    rid = str(pipeline_run_id).strip()
    ph = str(phase).strip()
    if not rid or not ph:
        raise ValueError("pipeline_run_id and phase must be non-empty")

    t = qualified_table(c, HITL_REVIEWS)
    q = f"""
    SELECT
      COUNT(1) AS n_total,
      COUNT_IF(status = 'approved') AS n_approved
    FROM {t}
    WHERE pipeline_run_id = {lit(rid)} AND phase = {lit(ph)}
    """
    row = _spark().sql(q).collect()[0]
    n_total = int(row["n_total"] if "n_total" in row else row[0])
    n_approved = int(row["n_approved"] if "n_approved" in row else row[1])
    if n_total == 0:
        return False
    return n_approved == n_total


def resolve_hitl(
    catalog: str,
    pipeline_run_id: str,
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
    rid = str(pipeline_run_id).strip()
    ph = str(phase).strip()
    at = str(artifact_type).strip()
    st = str(status).strip()
    rev = str(reviewer).strip() if reviewer is not None else None
    if not rid or not ph or not at or not st:
        raise ValueError("pipeline_run_id, phase, artifact_type, and status must be non-empty")
    if st not in ("approved", "rejected"):
        raise ValueError("status must be 'approved' or 'rejected'")

    t = qualified_table(c, HITL_REVIEWS)
    rev_sql = f"{lit(rev)}" if rev is not None and rev else "NULL"
    if rev is not None and rev:
        set_reviewer = f"reviewer = {lit(rev)}"
    else:
        set_reviewer = "reviewer = NULL"

    q = f"""
    UPDATE {t}
    SET
      status = {lit(st)},
      {set_reviewer},
      reviewed_at = current_timestamp()
    WHERE pipeline_run_id = {lit(rid)}
      AND phase = {lit(ph)}
      AND artifact_type = {lit(at)}
    """
    _spark().sql(q)
