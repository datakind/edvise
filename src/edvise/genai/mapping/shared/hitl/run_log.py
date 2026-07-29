"""
Append-only ``run_log.json`` for an institution — shared across IdentityAgent and SMA HITL.

Events are a union of :class:`RunEvent` (identity HITL) and :class:`SMARRunEvent` (SMA 2a HITL).
JSON on disk stays backward compatible: existing identity-only logs still validate.

Append-only ``mapping_override_log.json`` (same directory as ``run_log.json`` in an SMA run)
records :class:`MappingOverrideEvent` rows for post-gate manifest mapping overrides.
Additional override event models can be added later as a discriminated union on
``override_type`` without changing the manifest-mapping shape.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field

from edvise.genai.mapping.shared.hitl.json_io import (
    read_pydantic_json,
    write_pydantic_json,
)


class RunEvent(BaseModel):
    """
    One resolved IdentityAgent HITL item event.

    Written by the identity hitl resolver on ``resolve_items`` / ``apply_hook_spec``.
    Append-only — never mutated after writing.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str  # ISO datetime string
    resolved_by: str | None  # user identifier passed to resolver
    agent: str  # e.g. "identity_agent", "schema_mapping_agent"
    domain: str  # e.g. "grain", "term", "mapping"
    item_id: str
    choice: int
    option_id: str
    reentry: str  # "terminal" or "generate_hook"
    db_run_id: str | None = None
    task_run_id: str | None = None


class SMARRunEvent(BaseModel):
    """
    One resolved Schema Mapping Agent (2a) HITL item event.

    Appended to the same ``run_log.json`` as identity events; distinct fields from :class:`RunEvent`.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    resolved_by: str | None
    agent: str = "schema_mapping_agent"
    entity_type: str  # "cohort" or "course"
    item_id: str
    target_field: str
    failure_mode: str
    choice: int
    option_id: str
    reentry: str  # "terminal" or "direct_edit"
    db_run_id: str | None = None
    task_run_id: str | None = None


class RunLog(BaseModel):
    """
    Full audit trail for one institution across all agents and domains.

    Written to: institutions/<institution_id>/run_log.json
    One file per institution — pipeline stages append events.
    """

    model_config = ConfigDict(extra="forbid")

    institution_id: str
    events: list[RunEvent | SMARRunEvent] = Field(default_factory=list)


PipelineRunEvent = RunEvent | SMARRunEvent


class MappingOverrideEvent(BaseModel):
    """
    Audit row for a post-gate Schema Mapping Agent manifest mapping override.

    ``override_type`` is fixed for this model; future override kinds should use sibling
    event models and a union on ``override_type`` in :class:`MappingOverrideLog`.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    overridden_by: str
    agent: Literal["schema_mapping_agent"]
    override_type: Literal["manifest_mapping"]
    entity_type: Literal["cohort", "course"]
    target_field: str
    original_value: dict[str, Any]
    corrected_value: dict[str, Any]
    reviewer_notes: str | None
    rerun_scope: Literal["2b_full"]
    original_db_run_id: str
    original_task_run_id: str | None
    override_task_run_id: str | None


# When adding other override kinds, widen to a discriminated union, e.g.
# ``Annotated[MappingOverrideEvent | …, Field(discriminator="override_type")]``.
MappingOverrideLogEvent: TypeAlias = MappingOverrideEvent


class MappingOverrideLog(BaseModel):
    """
    Append-only mapping-override audit for one institution (SMA run directory).

    Written beside ``run_log.json`` under the schema-mapping-agent run root.
    """

    model_config = ConfigDict(extra="forbid")

    institution_id: str
    events: list[MappingOverrideLogEvent] = Field(default_factory=list)


def resolve_task_run_id() -> str | None:
    """
    Best-effort current Databricks *task* run id (workflow task), for run_log correlation.

    Tries env vars set on task clusters, then notebook/job context JSON. Never raises;
    returns ``None`` when not running on Databricks or when the id is unavailable.
    """
    for key in ("DATABRICKS_TASK_RUN_ID", "DATABRICKS_RUN_ID"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()

    try:
        from databricks.sdk.runtime import dbutils
    except Exception:
        return None

    try:
        # dbutils typing varies between local stubs and runtime; treat as Any for this
        # optional best-effort context read.
        dbutils_any = cast(Any, dbutils)
        ctx_json = (
            dbutils_any.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .toJson()
        )
        data = json.loads(ctx_json)
    except Exception:
        return None

    if isinstance(data.get("tags"), dict):
        tags = data["tags"]
        for k in ("taskRunId", "runId", "run_id"):
            val = tags.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()

    extra = data.get("extraContext")
    if isinstance(extra, dict):
        for k in ("taskRunId", "runId"):
            val = extra.get(k)
            if val is not None and str(val).strip():
                return str(val).strip()

    return None


def append_run_log_event(
    run_log_path: Path,
    institution_id: str,
    event: RunEvent | SMARRunEvent,
) -> None:
    """
    Append one event to ``run_log_path``. Creates the file if missing; never removes events.

    When ``event.task_run_id`` is unset, fills it from :func:`resolve_task_run_id` so local
    runs stay ``null`` and Databricks tasks record the task run id without threading it
    through every resolver.
    """
    if run_log_path.exists():
        run_log = read_pydantic_json(run_log_path, RunLog)
    else:
        run_log = RunLog(institution_id=institution_id)

    tid = resolve_task_run_id()
    if event.task_run_id is None and tid is not None:
        event = event.model_copy(update={"task_run_id": tid})

    run_log.events.append(event)
    write_pydantic_json(run_log_path, run_log)


def append_mapping_override_event(
    override_log_path: Path,
    institution_id: str,
    event: MappingOverrideLogEvent,
) -> None:
    """
    Append one event to ``override_log_path``. Creates the file if missing.

    When ``event.override_task_run_id`` is unset, fills it from :func:`resolve_task_run_id`
    (same behavior as :func:`append_run_log_event`).
    """
    if override_log_path.exists():
        override_log = read_pydantic_json(override_log_path, MappingOverrideLog)
    else:
        override_log = MappingOverrideLog(institution_id=institution_id)

    tid = resolve_task_run_id()
    if event.override_task_run_id is None and tid is not None:
        event = event.model_copy(update={"override_task_run_id": tid})

    override_log.events.append(event)
    write_pydantic_json(override_log_path, override_log)


__all__ = [
    "MappingOverrideEvent",
    "MappingOverrideLog",
    "MappingOverrideLogEvent",
    "PipelineRunEvent",
    "RunEvent",
    "RunLog",
    "SMARRunEvent",
    "append_mapping_override_event",
    "append_run_log_event",
    "resolve_task_run_id",
]
