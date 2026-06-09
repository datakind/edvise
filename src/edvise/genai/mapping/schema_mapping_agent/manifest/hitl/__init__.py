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
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver import (
    SMAHITLResolverError,
    apply_2a_manifest_repair,
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
    "apply_2a_manifest_repair",
    "HITL_CONFIDENCE_THRESHOLD",
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
