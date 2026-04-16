"""
Append-only ``run_log.json`` for an institution — shared across IdentityAgent and SMA HITL.

Events are a union of :class:`RunEvent` (identity HITL) and :class:`SMARRunEvent` (SMA 2a HITL).
JSON on disk stays backward compatible: existing identity-only logs still validate.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json, write_pydantic_json


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


def append_run_log_event(
    run_log_path: Path,
    institution_id: str,
    event: RunEvent | SMARRunEvent,
) -> None:
    """
    Append one event to ``run_log_path``. Creates the file if missing; never removes events.
    """
    if run_log_path.exists():
        run_log = read_pydantic_json(run_log_path, RunLog)
    else:
        run_log = RunLog(institution_id=institution_id)

    run_log.events.append(event)
    write_pydantic_json(run_log_path, run_log)


__all__ = [
    "PipelineRunEvent",
    "RunEvent",
    "RunLog",
    "SMARRunEvent",
    "append_run_log_event",
]
