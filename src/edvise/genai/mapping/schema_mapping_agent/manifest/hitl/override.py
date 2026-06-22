"""
Post-gate SMA manifest mapping override helpers.

Wraps :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver.apply_manifest_mapping_override`
for local disk paths and Unity Catalog volume paths. After an override, re-run SMA step 2b
(``rerun_scope="2b_full"`` in ``mapping_override_log.json``) to regenerate the transformation map.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver import (
    apply_manifest_mapping_override,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import FieldMappingRecord
from edvise.genai.mapping.shared.unity_volume_files import (
    is_unity_catalog_volume_path,
    read_unity_file_text,
    write_unity_file_text,
)


class ManifestOverrideError(Exception):
    """Raised when a manifest mapping override cannot be applied."""


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


def load_correction_json(path: str | Path) -> FieldMappingRecord:
    """Load a :class:`FieldMappingRecord` override from a JSON file."""
    p = Path(path)
    if not p.is_file():
        raise ManifestOverrideError(f"Override file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestOverrideError(f"Invalid JSON in override file {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestOverrideError(
            f"Override file must contain a JSON object: {p}"
        )
    try:
        return FieldMappingRecord.model_validate(data)
    except Exception as exc:
        raise ManifestOverrideError(
            f"Override file is not a valid FieldMappingRecord: {p}"
        ) from exc


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
    apply_manifest_mapping_override(
        Path(manifest_path),
        entity_type,
        target_field,
        corrected,
        override_log_path=Path(override_log_path),
        overridden_by=overridden_by,
        original_db_run_id=original_db_run_id,
        original_task_run_id=original_task_run_id,
        reviewer_notes=reviewer_notes,
        institution_id=institution_id,
    )


def _empty_mapping_override_log_json(institution_id: str) -> str:
    payload = {"institution_id": institution_id, "events": []}
    return json.dumps(payload, indent=2) + "\n"


def _read_volume_text_or_default(
    uc_path: str, *, default_text: str | None = None
) -> str:
    try:
        return read_unity_file_text(uc_path)
    except Exception as exc:
        if default_text is None:
            raise ManifestOverrideError(
                f"Could not read Unity Catalog file {uc_path!r}: {exc}"
            ) from exc
        return default_text


def override_manifest_mapping_on_volume(
    manifest_uc_path: str,
    entity_type: Literal["cohort", "course"],
    target_field: str,
    corrected: FieldMappingRecord | dict[str, Any],
    *,
    override_log_uc_path: str,
    overridden_by: str,
    original_db_run_id: str,
    original_task_run_id: str | None = None,
    reviewer_notes: str | None = None,
    institution_id: str | None = None,
) -> None:
    """
    Download ``manifest_map.json`` and ``mapping_override_log.json`` from UC volumes, apply the
    override, then upload both files back in place.
    """
    manifest_uc_path = str(manifest_uc_path).strip()
    override_log_uc_path = str(override_log_uc_path).strip()
    if not is_unity_catalog_volume_path(manifest_uc_path):
        raise ManifestOverrideError(
            f"manifest_uc_path must be a /Volumes/… path, got {manifest_uc_path!r}"
        )
    if not is_unity_catalog_volume_path(override_log_uc_path):
        raise ManifestOverrideError(
            f"override_log_uc_path must be a /Volumes/… path, got {override_log_uc_path!r}"
        )

    override_default: str | None = None
    if institution_id:
        override_default = _empty_mapping_override_log_json(institution_id)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        local_manifest = tmp / "manifest_map.json"
        local_override_log = tmp / "mapping_override_log.json"

        local_manifest.write_text(
            _read_volume_text_or_default(manifest_uc_path),
            encoding="utf-8",
        )
        local_override_log.write_text(
            _read_volume_text_or_default(
                override_log_uc_path, default_text=override_default
            ),
            encoding="utf-8",
        )

        override_manifest_mapping(
            local_manifest,
            entity_type,
            target_field,
            corrected,
            override_log_path=local_override_log,
            overridden_by=overridden_by,
            original_db_run_id=original_db_run_id,
            original_task_run_id=original_task_run_id,
            reviewer_notes=reviewer_notes,
            institution_id=institution_id,
        )

        write_unity_file_text(
            manifest_uc_path, local_manifest.read_text(encoding="utf-8")
        )
        write_unity_file_text(
            override_log_uc_path, local_override_log.read_text(encoding="utf-8")
        )


def override_manifest_mapping_at_path(
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
    Apply an override using either local paths or Unity Catalog ``/Volumes/…`` paths.

    When ``manifest_path`` is a volume path, ``override_log_path`` must also be a volume path.
    """
    manifest_str = str(manifest_path).strip()
    override_str = str(override_log_path).strip()
    if is_unity_catalog_volume_path(manifest_str):
        override_manifest_mapping_on_volume(
            manifest_str,
            entity_type,
            target_field,
            corrected,
            override_log_uc_path=override_str,
            overridden_by=overridden_by,
            original_db_run_id=original_db_run_id,
            original_task_run_id=original_task_run_id,
            reviewer_notes=reviewer_notes,
            institution_id=institution_id,
        )
        return

    override_manifest_mapping(
        manifest_str,
        entity_type,
        target_field,
        corrected,
        override_log_path=override_str,
        overridden_by=overridden_by,
        original_db_run_id=original_db_run_id,
        original_task_run_id=original_task_run_id,
        reviewer_notes=reviewer_notes,
        institution_id=institution_id,
    )


__all__ = [
    "ManifestOverrideError",
    "load_correction_json",
    "override_manifest_mapping",
    "override_manifest_mapping_at_path",
    "override_manifest_mapping_on_volume",
    "unmapped_field_mapping_record",
]
