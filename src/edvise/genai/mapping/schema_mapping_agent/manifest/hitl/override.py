"""
Post-gate SMA manifest mapping override helpers for the onboard job.

Wraps :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver.apply_manifest_mapping_override`
for local disk paths (including Databricks-mounted ``/Volumes/...``). After overrides, gate_2
continues with Step 2b (``rerun_scope="2b_full"`` in ``mapping_override_log.json``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver import (
    apply_manifest_mapping_override,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingRecord,
)


class ManifestOverrideError(Exception):
    """Raised when a manifest mapping override cannot be applied."""


@dataclass(frozen=True)
class ManifestMappingOverrideRequest:
    """One post-gate manifest field override within a batch."""

    entity_type: Literal["cohort", "course"]
    target_field: str
    corrected: FieldMappingRecord
    reviewer_notes: str | None = None


def unmapped_field_mapping_record(target_field: str) -> FieldMappingRecord:
    """
    Build a :class:`FieldMappingRecord` that marks ``target_field`` as unmapped.

    Sets ``source_column``, ``source_table``, ``join``, and ``row_selection`` to ``None``.
    """
    tf = str(target_field).strip()
    if not tf:
        raise ManifestOverrideError("target_field must be non-empty")
    return FieldMappingRecord.model_validate(
        {
            "target_field": tf,
            "source_column": None,
            "source_table": None,
            "join": None,
            "row_selection": None,
            "confidence": 1.0,
        }
    )


def _parse_override_entry(raw: Any, *, index: int) -> ManifestMappingOverrideRequest:
    if not isinstance(raw, dict):
        raise ManifestOverrideError(
            f"Override entry {index} must be a JSON object, got {type(raw).__name__}"
        )
    entity_type = raw.get("entity_type")
    if entity_type not in ("cohort", "course"):
        raise ManifestOverrideError(
            f"Override entry {index} requires entity_type 'cohort' or 'course'"
        )
    target_field = str(raw.get("target_field") or "").strip()
    if not target_field:
        raise ManifestOverrideError(f"Override entry {index} requires target_field")

    unmap = raw.get("unmap") is True
    correction = raw.get("correction")
    if unmap and correction is not None:
        raise ManifestOverrideError(
            f"Override entry {index}: set unmap=true or correction, not both"
        )
    if unmap:
        corrected = unmapped_field_mapping_record(target_field)
    elif correction is not None:
        try:
            corrected = FieldMappingRecord.model_validate(correction)
        except Exception as exc:
            raise ManifestOverrideError(
                f"Override entry {index} correction is not a valid FieldMappingRecord"
            ) from exc
        if corrected.target_field != target_field:
            raise ManifestOverrideError(
                f"Override entry {index}: correction.target_field "
                f"{corrected.target_field!r} must match target_field {target_field!r}"
            )
    else:
        raise ManifestOverrideError(
            f"Override entry {index} requires unmap=true or a correction object"
        )

    reviewer_notes = raw.get("reviewer_notes")
    notes = str(reviewer_notes).strip() if reviewer_notes is not None else None
    if notes == "":
        notes = None
    return ManifestMappingOverrideRequest(
        entity_type=entity_type,
        target_field=target_field,
        corrected=corrected,
        reviewer_notes=notes,
    )


def load_overrides_json(path: str | Path) -> list[ManifestMappingOverrideRequest]:
    """
    Load a batch of overrides from JSON.

    Accepts either ``{"overrides": [ ... ]}`` or a top-level array. Each entry::

        {
          "entity_type": "cohort",
          "target_field": "learner_id",
          "correction": { ... FieldMappingRecord ... },
          "reviewer_notes": "optional"
        }

    Or omit ``correction`` and set ``"unmap": true`` to leave the field unmapped.
    """
    p = Path(path)
    if not p.is_file():
        raise ManifestOverrideError(f"Overrides file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestOverrideError(
            f"Invalid JSON in overrides file {p}: {exc}"
        ) from exc

    if isinstance(data, dict):
        entries = data.get("overrides")
        if not isinstance(entries, list):
            raise ManifestOverrideError(
                f"Overrides file object must contain an 'overrides' array: {p}"
            )
    elif isinstance(data, list):
        entries = data
    else:
        raise ManifestOverrideError(
            f"Overrides file must be a JSON object or array: {p}"
        )
    if not entries:
        raise ManifestOverrideError(f"Overrides file contains no entries: {p}")
    return [_parse_override_entry(entry, index=i) for i, entry in enumerate(entries)]


def override_manifest_mappings(
    manifest_path: str | Path,
    overrides: list[ManifestMappingOverrideRequest],
    *,
    override_log_path: str | Path,
    overridden_by: str,
    original_db_run_id: str,
    original_task_run_id: str | None = None,
    institution_id: str | None = None,
) -> int:
    """
    Apply multiple manifest mapping overrides on local disk paths.

    Returns the number of overrides applied. Each override appends one audit event.
    """
    if not overrides:
        raise ManifestOverrideError("overrides must be non-empty")
    manifest = Path(manifest_path)
    log_path = Path(override_log_path)
    for req in overrides:
        apply_manifest_mapping_override(
            manifest,
            req.entity_type,
            req.target_field,
            req.corrected,
            override_log_path=log_path,
            overridden_by=overridden_by,
            original_db_run_id=original_db_run_id,
            original_task_run_id=original_task_run_id,
            reviewer_notes=req.reviewer_notes,
            institution_id=institution_id,
        )
    return len(overrides)


def override_manifest_mapping(
    manifest_path: str | Path,
    entity_type: Literal["cohort", "course"],
    target_field: str,
    corrected: FieldMappingRecord | dict[str, Any],
    *,
    override_log_path: str | Path,
    overridden_by: str,
    original_db_run_id: str,
    original_task_run_id: str | None = None,
    reviewer_notes: str | None = None,
    institution_id: str | None = None,
) -> None:
    """
    Apply a single manifest mapping override on local disk paths.

    See :func:`apply_manifest_mapping_override` for merge semantics (original ``confidence`` and
    ``rationale`` are preserved; ``review_status`` becomes ``corrected_by_override``).
    """
    override_manifest_mappings(
        manifest_path,
        [
            ManifestMappingOverrideRequest(
                entity_type=entity_type,
                target_field=target_field,
                corrected=(
                    corrected
                    if isinstance(corrected, FieldMappingRecord)
                    else FieldMappingRecord.model_validate(corrected)
                ),
                reviewer_notes=reviewer_notes,
            )
        ],
        override_log_path=override_log_path,
        overridden_by=overridden_by,
        original_db_run_id=original_db_run_id,
        original_task_run_id=original_task_run_id,
        institution_id=institution_id,
    )


__all__ = [
    "ManifestMappingOverrideRequest",
    "ManifestOverrideError",
    "load_overrides_json",
    "override_manifest_mapping",
    "override_manifest_mappings",
    "unmapped_field_mapping_record",
]
