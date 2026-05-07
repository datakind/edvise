"""Cross-agent HITL utilities (gate checks, JSON I/O, run log, shared exceptions)."""

from __future__ import annotations

from .confidence import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from .exceptions import HITLBlockingError
from .gate import raise_if_hitl_pending
from .json_io import read_pydantic_json, write_pydantic_json
from .run_log import (
    ManifestRepairEvent,
    PipelineRunEvent,
    RepairLog,
    RepairLogEvent,
    RunEvent,
    RunLog,
    SMARRunEvent,
    append_repair_event,
    append_run_log_event,
    resolve_task_run_id,
)
from .time import utc_now_iso

__all__ = [
    "HITLBlockingError",
    "PIPELINE_HITL_CONFIDENCE_THRESHOLD",
    "ManifestRepairEvent",
    "PipelineRunEvent",
    "RepairLog",
    "RepairLogEvent",
    "RunEvent",
    "RunLog",
    "SMARRunEvent",
    "append_repair_event",
    "append_run_log_event",
    "resolve_task_run_id",
    "raise_if_hitl_pending",
    "read_pydantic_json",
    "utc_now_iso",
    "write_pydantic_json",
]
