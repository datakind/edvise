"""
Cross-artifact validation: Step 2b transformation plans vs resolved Step 2a manifest.

Mirrors :mod:`edvise.genai.mapping.schema_mapping_agent.manifest.validation` — returns
structured errors rather than silently mutating LLM output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TransformationPlanValidationErrorCode(str, Enum):
    HOOK_REQUIRED_ON_UNMAPPED = "HOOK_REQUIRED_ON_UNMAPPED"
    REVIEW_REQUIRED_ON_UNMAPPED = "REVIEW_REQUIRED_ON_UNMAPPED"


class TransformationPlanValidationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: Literal["cohort", "course"]
    target_field: str
    error_code: TransformationPlanValidationErrorCode
    detail: str = Field(
        ...,
        description="Human-readable explanation for logs, retries, and HITL tooling.",
    )


def is_manifest_record_unmapped(record: dict[str, Any] | None) -> bool:
    """
    True when Step 2a left the target unmappable (no source column or table).

    Matches manifest eval null-mapping checks and field-executor skip logic.
    """
    if record is None:
        return True
    return record.get("source_column") is None and record.get("source_table") is None


def manifest_mapping_for_target(
    manifest_map: dict[str, Any],
    entity_type: Literal["cohort", "course"],
    target_field: str,
) -> dict[str, Any] | None:
    manifests = manifest_map.get("manifests")
    if not isinstance(manifests, dict):
        return None
    section = manifests.get(entity_type)
    if not isinstance(section, dict):
        return None
    mappings = section.get("mappings")
    if not isinstance(mappings, list):
        return None
    tf = str(target_field or "").strip()
    for rec in mappings:
        if not isinstance(rec, dict):
            continue
        if str(rec.get("target_field") or "").strip() == tf:
            return rec
    return None


def _plans_from_entity_section(entity_blob: object) -> list[dict[str, Any]] | None:
    if not isinstance(entity_blob, dict):
        return None
    raw_plans = entity_blob.get("plans")
    if not isinstance(raw_plans, list):
        return None
    return [p for p in raw_plans if isinstance(p, dict)]


def validate_transformation_plans_against_manifest(
    transformation_data: dict[str, Any],
    manifest_map: dict[str, Any],
) -> list[TransformationPlanValidationError]:
    """
    Check each plan against its resolved manifest FieldMappingRecord.

    Gate 1 ``leave_unmapped`` targets must not carry Step 2b hook or review routing —
    there is no source Series for utilities or transform hooks to consume.
    """
    errors: list[TransformationPlanValidationError] = []
    tmaps = transformation_data.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return errors

    for entity_type in ("cohort", "course"):
        plans = _plans_from_entity_section(tmaps.get(entity_type))
        if not plans:
            continue
        for plan in plans:
            tf = str(plan.get("target_field") or "").strip()
            if not tf:
                continue
            mrec = manifest_mapping_for_target(manifest_map, entity_type, tf)
            if not is_manifest_record_unmapped(mrec):
                continue

            if plan.get("hook_required") is True:
                errors.append(
                    TransformationPlanValidationError(
                        entity_type=entity_type,
                        target_field=tf,
                        error_code=TransformationPlanValidationErrorCode.HOOK_REQUIRED_ON_UNMAPPED,
                        detail=(
                            f"{entity_type}.{tf}: manifest source_column and source_table are "
                            "both null (Gate 1 leave_unmapped) but plan has hook_required=true. "
                            "Use empty steps only; omit hook_required and review_required."
                        ),
                    )
                )
            if plan.get("review_required") is True:
                errors.append(
                    TransformationPlanValidationError(
                        entity_type=entity_type,
                        target_field=tf,
                        error_code=TransformationPlanValidationErrorCode.REVIEW_REQUIRED_ON_UNMAPPED,
                        detail=(
                            f"{entity_type}.{tf}: manifest is unmapped but plan has "
                            "review_required=true. Unmappable fields are decided at Step 2a only."
                        ),
                    )
                )

    return errors


def raise_pydantic_validation_error_if_any(
    errors: list[TransformationPlanValidationError],
) -> None:
    """
    Raise :class:`ValidationError` so Step 2b ``llm_complete_with_parse_retry`` can retry.

    One pydantic error entry per validation row (entity + target_field in ``loc``).
    """
    if not errors:
        return
    raise ValidationError.from_exception_data(
        "TransformationPlanManifestValidation",
        [
            {
                "type": "value_error",
                "loc": (e.entity_type, e.target_field, e.error_code.value),
                "input": None,
                "ctx": {"error": ValueError(e.detail)},
            }
            for e in errors
        ],
    )


__all__ = [
    "TransformationPlanValidationError",
    "TransformationPlanValidationErrorCode",
    "is_manifest_record_unmapped",
    "manifest_mapping_for_target",
    "raise_pydantic_validation_error_if_any",
    "validate_transformation_plans_against_manifest",
]
