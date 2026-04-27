"""
Derive Databricks / OpenAI ``json_schema`` bodies from Pydantic models, then post-process
(:func:`~edvise.genai.mapping.shared.schema_utils.inline_json_schema_refs` for ``$ref``;
optional ``if``/``then`` for Identity grain HITL rules) in one place.

Schema Mapping Agent (SMA) Step 2a/2b and refinement pass shapes are defined here as
loose-but-structured Pydantic models so the gateway matches parse-time types without
importing heavy manifest modules (avoids cycles).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermContract


# ---------------------------------------------------------------------------
# Identity grain — one LLM object: GrainContract + top-level hitl_items
# ---------------------------------------------------------------------------


class IdentityGrainLlmResponse(GrainContract):
    """
    LLM return shape for one dataset grain call (validates in parse; gateway schema is derived
    from this model).
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    hitl_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured HITL; validated to HITLItem after parse.",
    )


# ---------------------------------------------------------------------------
# Identity term batch — envelope + per-dataset rows with always-empty nested hitl_items
# ---------------------------------------------------------------------------


class TermDatasetLlmRow(TermContract):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    hitl_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per prompt must be []; HITL lives on envelope hitl_items.",
    )


class IdentityTermBatchLlmResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    institution_id: str
    datasets: dict[str, TermDatasetLlmRow]
    hitl_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Canonical HITL list for the whole batch response.",
    )


# ---------------------------------------------------------------------------
# Identity hook generation — minimal HookSpec-shaped object for the gateway
# ---------------------------------------------------------------------------


class IdentityHookGenFunctionRow(BaseModel):
    """Slim function row: gateway allows extra keys on each function in practice."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="allow",
    )
    name: str
    description: str
    draft: str | None = None


class IdentityHookSpecLlmOut(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    file: str | None = None
    functions: list[IdentityHookGenFunctionRow] = Field(
        ...,
        description="Hook function drafts; extra keys on each function are allowed in practice.",
    )


# ---------------------------------------------------------------------------
# Post-process: grain HITL when hitl_flag is true
# ---------------------------------------------------------------------------


def _loose_object_items() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": True,
    }


def _grain_allof_hitl_if_then() -> dict[str, Any]:
    return {
        "if": {
            "type": "object",
            "properties": {"hitl_flag": {"const": True}},
            "required": ["hitl_flag"],
        },
        "then": {
            "type": "object",
            "properties": {
                "hitl_items": {
                    "type": "array",
                    "minItems": 1,
                    "items": _loose_object_items(),
                }
            },
            "required": ["hitl_items"],
        },
    }


def build_identity_grain_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    raw = IdentityGrainLlmResponse.model_json_schema()
    body = inline_json_schema_refs(raw)
    return {
        "allOf": [
            body,
            _grain_allof_hitl_if_then(),
        ],
    }


def build_identity_term_batch_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(IdentityTermBatchLlmResponse.model_json_schema())


def build_identity_hook_spec_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(IdentityHookSpecLlmOut.model_json_schema())


# ---------------------------------------------------------------------------
# Schema Mapping Agent (SMA) — Step 2a envelope, 2a entity, 2b, refinement
# ---------------------------------------------------------------------------


class SmaFieldMappingFragment(BaseModel):
    """Per-entity manifest body (loose `mappings` / `column_aliases` — validated in pipeline)."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    entity_type: str
    target_schema: str
    mappings: list[dict[str, Any]]
    column_aliases: list[dict[str, Any]]


class SmaMappingManifestsCohortCourse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cohort: SmaFieldMappingFragment
    course: SmaFieldMappingFragment


class SmaMappingManifestEnvelopeGateway(BaseModel):
    """
    Batched Step 2a top-level object. `merge_step2a` may add ``institution_id`` / ``pipeline_version``;
    only ``manifests`` is required in the hand JSON schema, so the rest are optional.
    """

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    institution_id: str | None = None
    pipeline_version: str | None = None
    manifests: SmaMappingManifestsCohortCourse


class SmaStep2aEntityPassResponse(RootModel[dict[str, Any]]):
    """
    One cohort or course sub-pass: any JSON object
    (``merge_step2a_entity_manifests`` accepts several fragment shapes).
    """


class SmaTransformationMapSection(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    entity_type: str
    target_schema: str
    plans: list[dict[str, Any]]


class SmaTransformationMapsCohortCourse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cohort: SmaTransformationMapSection
    course: SmaTransformationMapSection


class SmaTransformationMapWrapperGateway(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    institution_id: str | None = None
    transformation_maps: SmaTransformationMapsCohortCourse


class SmaRefinementPass1Gateway(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    field_statuses: dict[str, Any]
    refined_corrections: dict[str, Any]
    hitl_flags: list[dict[str, Any]]


class SmaRefinementPass2Gateway(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    items: list[dict[str, Any]]


def build_sma_mapping_manifest_envelope_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(
        SmaMappingManifestEnvelopeGateway.model_json_schema()
    )


def build_sma_step2a_entity_pass_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(SmaStep2aEntityPassResponse.model_json_schema())


def build_sma_transformation_map_wrapper_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(
        SmaTransformationMapWrapperGateway.model_json_schema()
    )


def build_sma_refinement_pass1_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(SmaRefinementPass1Gateway.model_json_schema())


def build_sma_refinement_pass2_gateway_json_schema() -> dict[str, Any]:
    from edvise.genai.mapping.shared.schema_utils import inline_json_schema_refs

    return inline_json_schema_refs(SmaRefinementPass2Gateway.model_json_schema())