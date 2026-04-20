"""
Schema Mapping Agent (2a) HITL on-disk helpers.

Gate (:func:`check_sma_hitl_gate`) and apply (:func:`resolve_sma_items`) live here;
Pydantic models are in :mod:`edvise.genai.mapping.schema_mapping_agent.hitl.schemas`.
IdentityAgent equivalents: :mod:`edvise.genai.mapping.identity_agent.hitl.resolver`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import load_sma_hitl
from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
    SMAHITLItem,
    SMAReentryDepth,
    add_alias_if_missing,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    EntityType,
    FieldMappingManifest,
    FieldMappingRecord,
    MappingManifestEnvelope,
    ReviewStatus,
)
from edvise.genai.mapping.shared.hitl import raise_if_hitl_pending
from edvise.genai.mapping.shared.hitl.json_io import write_pydantic_json
from edvise.genai.mapping.shared.hitl.run_log import SMARRunEvent, append_run_log_event
from edvise.genai.mapping.shared.hitl.time import utc_now_iso

logger = logging.getLogger(__name__)


class SMAHITLResolverError(Exception):
    """Raised when SMA HITL resolution cannot apply (e.g. missing target_field)."""


def check_sma_hitl_gate(hitl_path: str | Path) -> None:
    """
    Raises :class:`~edvise.genai.mapping.shared.hitl.HITLBlockingError` if any SMA
    HITL items are still pending (no ``choice``, or ``direct_edit`` without mapping).

    Intended to run before downstream pipeline steps (e.g. Step 2b) on every run;
    there is no optional or execution-mode bypass for this check.

    Prints and returns cleanly when the gate passes. Does not mutate files.
    """
    path = Path(hitl_path)
    envelope = load_sma_hitl(path)

    if not envelope.items:
        print("✓ No SMA HITL items — pipeline gate clear.")
        return

    if envelope.is_clear:
        print(f"✓ SMA HITL gate clear — {len(envelope.items)} item(s) reviewed.")
        return

    def _format_item(i: SMAHITLItem) -> str:
        if i.choice is None:
            return f"[{i.item_id}] {i.target_field} — {i.hitl_question[:80]}..."
        return (
            f"[{i.item_id}] {i.target_field} — direct_edit selected; "
            "populate direct_edit_field_mapping."
        )

    raise_if_hitl_pending(
        envelope.gate_pending,
        hitl_path=path,
        format_item=_format_item,
        instructions=(
            "  • Set 'choice' to the 1-based index of your selected option "
            "(1 … number of options), or populate direct_edit_field_mapping "
            "if you chose direct_edit.\n"
            "  • Re-run this cell."
        ),
    )


def _load_manifest_for_entity(
    manifest_path: Path,
    entity_type: Literal["cohort", "course"],
) -> tuple[MappingManifestEnvelope | None, FieldMappingManifest, EntityType]:
    """
    Load either a :class:`MappingManifestEnvelope` (notebook / pipeline layout) or a
    single-entity :class:`FieldMappingManifest` (``sma_manifest_output.json``).
    """
    raw = json.loads(manifest_path.read_text())
    et = EntityType(entity_type)
    if "manifests" in raw:
        env = MappingManifestEnvelope.model_validate(raw)
        if et not in env.manifests:
            raise SMAHITLResolverError(
                f"Manifest envelope has no {entity_type!r} entry: {manifest_path}"
            )
        return env, env.manifests[et], et
    fm = FieldMappingManifest.model_validate(raw)
    if fm.entity_type != et:
        raise SMAHITLResolverError(
            f"Manifest entity_type {fm.entity_type!r} does not match HITL entity_type "
            f"{entity_type!r} ({manifest_path})"
        )
    return None, fm, et


def _save_manifest_for_entity(
    manifest_path: Path,
    envelope: MappingManifestEnvelope | None,
    fm: FieldMappingManifest,
    entity_type: EntityType,
) -> None:
    if envelope is not None:
        envelope.manifests[entity_type] = fm
        write_pydantic_json(manifest_path, envelope)
    else:
        write_pydantic_json(manifest_path, fm)


def _mapping_index_for_target(
    mappings: list[FieldMappingRecord], target_field: str
) -> int:
    for i, m in enumerate(mappings):
        if m.target_field == target_field:
            return i
    raise SMAHITLResolverError(
        f"No manifest mapping with target_field={target_field!r} "
        f"(HITL item references a field absent from this entity manifest)."
    )


def resolve_sma_items(
    hitl_path: str | Path,
    manifest_path: str | Path,
    resolved_by: str | None = None,
    run_log_path: str | Path | None = None,
) -> int:
    """
    Apply reviewer selections from ``sma_hitl*.json`` into the mapping manifest.

    For each :class:`SMAHITLItem` with a resolvable selection (``choice`` set and
    :meth:`SMAHITLItem.resolved_field_mapping` not ``None``):

    - Replaces ``manifest.mappings[...]`` for ``target_field`` with the resolved
      :class:`FieldMappingRecord`, setting ``review_status=corrected_by_hitl`` and
      ``reviewer_notes`` from the item when provided.
    - Appends optional ``column_alias`` from the selected TERMINAL option via
      :func:`add_alias_if_missing`.
    - Appends a :class:`~edvise.genai.mapping.shared.hitl.run_log.SMARRunEvent` when
      ``run_log_path`` is set.

    ``manifest_path`` may be a full :class:`MappingManifestEnvelope` JSON (keys
    ``schema_version``, ``institution_id``, ``manifests``) or a single
    :class:`FieldMappingManifest` (e.g. ``sma_manifest_output.json``). The HITL
    envelope's ``entity_type`` selects which slice to update.

    Parameters
    ----------
    hitl_path
        Path to ``sma_hitl.json`` / ``sma_hitl_cohort.json`` / ``sma_hitl_course.json``.
    manifest_path
        Path to the mapping manifest to update in place.
    resolved_by
        Optional identifier for audit (e.g. reviewer id).
    run_log_path
        Optional path to ``run_log.json`` (created if missing).

    Returns
    -------
    int
        Number of HITL items applied.
    """
    hitl_path = Path(hitl_path)
    manifest_path = Path(manifest_path)

    hitl_envelope = load_sma_hitl(hitl_path)
    env_wrapper, fm, entity_key = _load_manifest_for_entity(
        manifest_path, hitl_envelope.entity_type
    )

    institution_id = (
        env_wrapper.institution_id
        if env_wrapper is not None
        else hitl_envelope.institution_id
    )
    if env_wrapper is not None and env_wrapper.institution_id != hitl_envelope.institution_id:
        raise SMAHITLResolverError(
            f"HITL institution_id {hitl_envelope.institution_id!r} does not match "
            f"manifest institution_id {env_wrapper.institution_id!r}"
        )

    applied_count = 0

    for item in hitl_envelope.items:
        if item.choice is None:
            continue
        resolved = item.resolved_field_mapping
        if resolved is None:
            logger.debug(
                "Skipping HITL item %s — no resolved_field_mapping yet (direct_edit?)",
                item.item_id,
            )
            continue

        selected = item.selected_option()
        if selected is None:
            continue

        idx = _mapping_index_for_target(fm.mappings, item.target_field)

        updated = resolved.model_copy(
            update={
                "review_status": ReviewStatus.corrected_by_hitl,
                "reviewer_notes": item.reviewer_note,
            }
        )
        fm.mappings[idx] = updated

        if selected.reentry == SMAReentryDepth.TERMINAL and selected.column_alias is not None:
            add_alias_if_missing(fm, selected.column_alias)

        if run_log_path is not None:
            assert item.choice is not None
            event = SMARRunEvent(
                timestamp=utc_now_iso(),
                resolved_by=resolved_by,
                agent="schema_mapping_agent",
                entity_type=item.entity_type,
                item_id=item.item_id,
                target_field=item.target_field,
                failure_mode=item.failure_mode.value,
                choice=item.choice,
                option_id=selected.option_id,
                reentry=selected.reentry.value,
            )
            append_run_log_event(Path(run_log_path), institution_id, event)

        applied_count += 1
        logger.info(
            "SMA HITL applied item_id=%s target_field=%s option_id=%s",
            item.item_id,
            item.target_field,
            selected.option_id,
        )

    _save_manifest_for_entity(manifest_path, env_wrapper, fm, entity_key)
    print(
        f"✓ SMA HITL: applied {applied_count} item(s); manifest updated at {manifest_path.name}."
    )
    return applied_count


__all__ = ["SMAHITLResolverError", "check_sma_hitl_gate", "resolve_sma_items"]
