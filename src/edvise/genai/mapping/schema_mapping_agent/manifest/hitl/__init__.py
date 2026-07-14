"""Schema Mapping Agent Step 2a — manifest HITL payloads, artifacts, and resolver."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.artifacts import (
    SMA_HITL_BASENAME,
    SMA_MANIFEST_OUTPUT_BASENAME,
    load_sma_hitl,
    load_sma_manifest_output,
    unique_sma_hitl_items_by_item_id,
    write_sma_hitl_artifact,
    write_sma_hitl_and_manifest_artifacts,
    write_sma_manifest_artifact,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.override import (
    ManifestMappingOverrideRequest,
    ManifestOverrideError,
    load_overrides_json,
    override_manifest_mapping,
    override_manifest_mappings,
    unmapped_field_mapping_record,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver import (
    SMAHITLResolverError,
    apply_manifest_mapping_override,
    check_sma_hitl_gate,
    resolve_sma_items,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.schemas import (
    HITL_CONFIDENCE_THRESHOLD,
    InstitutionSMAHITLItems,
    SMAFailureMode,
    SMAHITLItem,
    SMAHITLOption,
    SMAReentryDepth,
    SMARRunEvent,
    add_alias_if_missing,
)

__all__ = [
    "apply_manifest_mapping_override",
    "HITL_CONFIDENCE_THRESHOLD",
    "ManifestMappingOverrideRequest",
    "ManifestOverrideError",
    "load_overrides_json",
    "override_manifest_mapping",
    "override_manifest_mappings",
    "unmapped_field_mapping_record",
    "InstitutionSMAHITLItems",
    "SMAFailureMode",
    "SMAHITLItem",
    "SMAHITLOption",
    "SMAHITLResolverError",
    "SMAReentryDepth",
    "SMA_HITL_BASENAME",
    "SMA_MANIFEST_OUTPUT_BASENAME",
    "SMARRunEvent",
    "add_alias_if_missing",
    "check_sma_hitl_gate",
    "load_sma_hitl",
    "load_sma_manifest_output",
    "resolve_sma_items",
    "unique_sma_hitl_items_by_item_id",
    "write_sma_hitl_artifact",
    "write_sma_manifest_artifact",
    "write_sma_hitl_and_manifest_artifacts",
]
