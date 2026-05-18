"""Schema Mapping Agent Step 2b — transformation map HITL (review + hook preview / hook_required)."""

from __future__ import annotations

from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.gates import (
    check_transformation_hook_hitl_gate,
    check_transformation_review_hitl_gate,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation import (
    generate_sma_transform_hook_preview_rows_for_entity,
    load_hook_specs_from_sma_preview_path,
    write_sma_transform_hook_preview_json,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_required_hitl import (
    apply_transformation_hook_hitl_resolutions,
    build_transformation_hook_hitl_envelope_for_entity,
    default_transformation_hook_hitl_options,
    iter_hook_required_plans,
    write_transformation_hook_hitl_envelope,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.review_hitl import (
    apply_transformation_review_resolutions,
    build_transformation_review_hitl_file_for_entity,
    write_transformation_review_hitl_file,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas import (
    InstitutionSMATransformationHookHITLItems,
    SMATransformationHookHITLItem,
    SMATransformationHookHITLOption,
    SMATransformationHookResolution,
    TransformationReviewHITLFile,
)

__all__ = [
    "InstitutionSMATransformationHookHITLItems",
    "SMATransformationHookHITLItem",
    "SMATransformationHookHITLOption",
    "SMATransformationHookResolution",
    "TransformationReviewHITLFile",
    "apply_transformation_hook_hitl_resolutions",
    "apply_transformation_review_resolutions",
    "build_transformation_hook_hitl_envelope_for_entity",
    "build_transformation_review_hitl_file_for_entity",
    "check_transformation_hook_hitl_gate",
    "check_transformation_review_hitl_gate",
    "default_transformation_hook_hitl_options",
    "generate_sma_transform_hook_preview_rows_for_entity",
    "iter_hook_required_plans",
    "load_hook_specs_from_sma_preview_path",
    "write_sma_transform_hook_preview_json",
    "write_transformation_hook_hitl_envelope",
    "write_transformation_review_hitl_file",
]
