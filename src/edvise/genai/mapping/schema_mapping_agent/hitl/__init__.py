"""Schema Mapping Agent (2a) HITL — review payloads, artifacts, and gate helpers."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import (
    SMA_HITL_BASENAME,
    SMA_MANIFEST_OUTPUT_BASENAME,
    load_sma_hitl,
    load_sma_manifest_output,
    unique_sma_hitl_items_by_item_id,
    write_sma_hitl_artifact,
    write_sma_manifest_artifact,
    write_sma_hitl_and_manifest_artifacts,
)
from edvise.genai.mapping.schema_mapping_agent.hitl.resolver import (
    SMAHITLResolverError,
    check_sma_hitl_gate,
    resolve_sma_items,
)
from edvise.genai.mapping.schema_mapping_agent.hitl.transformation_hook_hitl import (
    InstitutionSMATransformationHookHITLItems,
    SMATransformationHookHITLItem,
    SMATransformationHookHITLOption,
    SMATransformationHookResolution,
    apply_transformation_hook_hitl_resolutions,
    build_transformation_hook_hitl_envelope_for_entity,
    check_transformation_hook_hitl_gate,
    default_transformation_hook_hitl_options,
    write_transformation_hook_hitl_envelope,
)
from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
    HITL_CONFIDENCE_THRESHOLD,
    InstitutionSMAHITLItems,
    SMAFailureMode,
    SMAHITLItem,
    SMAHITLOption,
    SMAReentryDepth,
    SMARRunEvent,
    add_alias_if_missing,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import ColumnAlias

__all__ = [
    "ColumnAlias",
    "HITL_CONFIDENCE_THRESHOLD",
    "InstitutionSMATransformationHookHITLItems",
    "InstitutionSMAHITLItems",
    "SMAFailureMode",
    "SMATransformationHookHITLItem",
    "SMATransformationHookHITLOption",
    "SMATransformationHookResolution",
    "apply_transformation_hook_hitl_resolutions",
    "build_transformation_hook_hitl_envelope_for_entity",
    "check_transformation_hook_hitl_gate",
    "default_transformation_hook_hitl_options",
    "write_transformation_hook_hitl_envelope",
    "SMAHITLItem",
    "SMAHITLOption",
    "SMAReentryDepth",
    "SMA_HITL_BASENAME",
    "SMA_MANIFEST_OUTPUT_BASENAME",
    "SMARRunEvent",
    "add_alias_if_missing",
    "SMAHITLResolverError",
    "check_sma_hitl_gate",
    "resolve_sma_items",
    "load_sma_hitl",
    "load_sma_manifest_output",
    "unique_sma_hitl_items_by_item_id",
    "write_sma_hitl_artifact",
    "write_sma_manifest_artifact",
    "write_sma_hitl_and_manifest_artifacts",
]
