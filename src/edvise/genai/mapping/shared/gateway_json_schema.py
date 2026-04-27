"""
Databricks Foundation Model API JSON-schema response_format helpers.

The MLflow / OpenAI-compatible gateway used by edvise supports::

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": str, "strict": bool, "schema": <JSON Schema>},
    }

Response envelopes are built with :func:`~edvise.genai.mapping.shared.schema_utils.to_gateway_schema_from_dict`
(inlines ``$defs`` / ``$ref`` when present). We keep schemas small where needed for Databricks
JSON Schema subset limitations (some combinators). IdentityAgent **grain** uses a closed
:class:`GrainContract` (``additionalProperties: false``) and a typed ``dedup_policy``
matching :class:`DedupPolicy`. **Term batch** per-dataset objects match :class:`TermContract`
so the gateway cannot accept profile-input-shaped JSON in ``datasets``.

Opt-out: set env ``EDVISE_GENAI_JSON_SCHEMA=0`` to disable and fall back to plain
JSON prompting + ``strip_json_fences`` parsing.
"""

from __future__ import annotations

import os
from typing import Any

from edvise.genai.mapping.shared.schema_utils import to_gateway_schema_from_dict

_EDVISE_GENAI_JSON_SCHEMA_ENV = "EDVISE_GENAI_JSON_SCHEMA"


def genai_json_schema_enabled() -> bool:
    v = (os.environ.get(_EDVISE_GENAI_JSON_SCHEMA_ENV) or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


# --- Step 2a (MappingManifestEnvelope) ---------------------------------------


def mapping_manifest_envelope_response_format() -> dict[str, Any]:
    """
    Top-level mapping manifest JSON written by Step 2a (envelope; pipeline may inject ids).
    """
    entity = {
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "target_schema": {"type": "string"},
            "mappings": {"type": "array", "items": {"type": "object"}},
            "column_aliases": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["entity_type", "target_schema", "mappings", "column_aliases"],
        "additionalProperties": True,
    }
    return to_gateway_schema_from_dict(
        "mapping_manifest_envelope",
        {
            "type": "object",
            "properties": {
                "institution_id": {"type": "string"},
                "pipeline_version": {"type": "string"},
                "manifests": {
                    "type": "object",
                    "properties": {
                        "cohort": entity,
                        "course": entity,
                    },
                    "required": ["cohort", "course"],
                    "additionalProperties": False,
                },
            },
            "required": ["manifests"],
            "additionalProperties": True,
        },
        strict=False,
    )


def step2a_entity_pass_response_format() -> dict[str, Any]:
    """
    One Step 2a per-entity gateway call (cohort pass or course pass).

    ``merge_step2a_entity_manifests`` accepts either a top-level ``manifests`` fragment
    with a single entity key, *or* a :class:`FieldMappingManifest` top-level object, so
    the JSON schema only enforces a JSON object.
    """
    return to_gateway_schema_from_dict(
        "step2a_entity_pass",
        {"type": "object", "additionalProperties": True},
        strict=False,
    )


# --- Step 2b (transformation map wrapper) --------------------------------------


def transformation_map_wrapper_response_format() -> dict[str, Any]:
    section = {
        "type": "object",
        "properties": {
            "entity_type": {"type": "string"},
            "target_schema": {"type": "string"},
            "plans": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["entity_type", "target_schema", "plans"],
        "additionalProperties": True,
    }
    return to_gateway_schema_from_dict(
        "transformation_map_wrapper",
        {
            "type": "object",
            "properties": {
                "institution_id": {"type": "string"},
                "transformation_maps": {
                    "type": "object",
                    "properties": {
                        "cohort": section,
                        "course": section,
                    },
                    "required": ["cohort", "course"],
                    "additionalProperties": False,
                },
            },
            "required": ["transformation_maps"],
            "additionalProperties": True,
        },
        strict=False,
    )


# --- IdentityAgent: grain (GrainContract) ------------------------------------


def _ia_grain_dedup_hook_function_spec() -> dict[str, Any]:
    # :class:`HookFunctionSpec` — ``additionalProperties: false`` to match the model
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "signature": {"type": ["string", "null"]},
            "example_input": {"type": ["string", "null"]},
            "example_output": {"type": ["string", "number", "null"]},
            "draft": {"type": ["string", "null"]},
        },
        "required": ["name", "description"],
        "additionalProperties": False,
    }


def _ia_grain_dedup_hook_spec() -> dict[str, Any]:
    return {
        "type": ["object", "null"],
        "properties": {
            "file": {"type": ["string", "null"]},
            "functions": {
                "type": "array",
                "items": _ia_grain_dedup_hook_function_spec(),
            },
        },
        "required": ["functions"],
        "additionalProperties": False,
    }


def _ia_grain_dedup_policy() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "enum": [
                    "true_duplicate",
                    "temporal_collapse",
                    "categorical_priority",
                    "suffix_identifier",
                    "no_dedup",
                    "policy_required",
                ],
            },
            "sort_by": {"type": ["string", "null"]},
            "sort_ascending": {"type": ["boolean", "null"]},
            "keep": {"enum": [None, "first", "last"]},
            "suffix_column": {"type": ["string", "null"]},
            "priority_column": {"type": ["string", "null"]},
            "priority_order": {
                "type": ["array", "null"],
                "items": {"type": "string"},
            },
            "notes": {"type": "string"},
            "hook_spec": _ia_grain_dedup_hook_spec(),
        },
        "required": ["strategy"],
        "additionalProperties": False,
    }


def identity_grain_contract_response_format() -> dict[str, Any]:
    return to_gateway_schema_from_dict(
        "identity_grain_contract",
        {
            "type": "object",
            "properties": {
                "institution_id": {"type": "string"},
                "table": {"type": "string"},
                "learner_id_alias": {"type": ["string", "null"]},
                "post_clean_primary_key": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "dedup_policy": _ia_grain_dedup_policy(),
                "row_selection_required": {"type": "boolean"},
                "join_keys_for_2a": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "hitl_flag": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": [
                "institution_id",
                "table",
                "post_clean_primary_key",
                "dedup_policy",
                "row_selection_required",
                "join_keys_for_2a",
                "confidence",
                "hitl_flag",
                "reasoning",
            ],
            "additionalProperties": False,
        },
        strict=False,
    )


# --- IdentityAgent: term batch (InstitutionTermContract) ----------------------


def _ia_term_batch_hook_function_spec() -> dict[str, Any]:
    # Mirror HookFunctionSpec; example_output is str|int|float|None
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "signature": {"type": ["string", "null"]},
            "example_input": {"type": ["string", "null"]},
            "example_output": {"type": ["string", "number", "null"]},
            "draft": {"type": ["string", "null"]},
        },
        "required": ["name", "description"],
        "additionalProperties": True,
    }


def _ia_term_batch_hook_spec_value() -> dict[str, Any]:
    """
    Object or null, matching :class:`HookSpec` in term ``term_config``;
    not wrapped in a redundant outer ``{type: object}`` for nesting under ``type: []``.
    """
    return {
        "type": ["object", "null"],
        "properties": {
            "file": {"type": ["string", "null"]},
            "functions": {
                "type": "array",
                "items": _ia_term_batch_hook_function_spec(),
            },
        },
        "required": ["functions"],
        "additionalProperties": True,
    }


def _ia_term_batch_term_order_config_properties() -> dict[str, Any]:
    """
    Property map for a non-null :class:`TermOrderConfig`. Only ``term_extraction`` is required;
    other fields default in Pydantic. ``hook_spec`` is optional; use null or omit.
    """
    return {
        "term_col": {"type": ["string", "null"]},
        "year_col": {"type": ["string", "null"]},
        "season_col": {"type": ["string", "null"]},
        "season_map": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "raw": {"type": "string"},
                    "canonical": {"type": "string"},
                },
                "required": ["raw", "canonical"],
                "additionalProperties": True,
            },
        },
        "exclude_tokens": {
            "type": "array",
            "items": {"type": "string"},
        },
        "term_extraction": {
            "type": "string",
            "enum": ["standard", "hook_required"],
        },
        "hook_spec": _ia_term_batch_hook_spec_value(),
    }


def _ia_term_batch_term_config_property() -> dict[str, Any]:
    """``TermOrderConfig | None`` — use ``type: [object, null]`` (no top-level anyOf)."""
    props = _ia_term_batch_term_order_config_properties()
    return {
        "type": ["object", "null"],
        "properties": props,
        "required": ["term_extraction"],
        "additionalProperties": True,
    }


def _ia_term_batch_per_dataset_term_contract() -> dict[str, Any]:
    """
    :class:`TermContract` (plus optional ``hitl_items``; ignored by Pydantic, allowed by prompt).
    ``additionalProperties: false`` so profile-only keys (e.g. ``row_selection_required``) are rejected
    at the gateway before Pydantic sees them.
    """
    return {
        "type": "object",
        "properties": {
            "institution_id": {"type": "string"},
            "table": {"type": "string"},
            "term_config": _ia_term_batch_term_config_property(),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "hitl_flag": {"type": "boolean"},
            "reasoning": {"type": "string"},
            "hitl_items": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "required": [
            "institution_id",
            "table",
            "term_config",
            "confidence",
            "hitl_flag",
            "reasoning",
        ],
        "additionalProperties": False,
    }


def identity_term_batch_envelope_response_format() -> dict[str, Any]:
    # Aligns with InstitutionTermContract + per-dataset TermContract; top-level hitl_items is
    # produced by the model, stripped before Pydantic — keep optional in schema.
    return to_gateway_schema_from_dict(
        "identity_term_batch_envelope",
        {
            "type": "object",
            "properties": {
                "institution_id": {"type": "string"},
                "datasets": {
                    "type": "object",
                    "additionalProperties": _ia_term_batch_per_dataset_term_contract(),
                },
                "hitl_items": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                },
            },
            "required": ["institution_id", "datasets"],
            "additionalProperties": True,
        },
        strict=False,
    )


# --- IdentityAgent: hook generation (HookSpec) -----------------------------


def identity_hook_spec_response_format() -> dict[str, Any]:
    return to_gateway_schema_from_dict(
        "identity_hook_spec",
        {
            "type": "object",
            "properties": {
                "file": {"type": ["string", "null"]},
                "functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "draft": {"type": ["string", "null"]},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["functions"],
            "additionalProperties": True,
        },
        strict=False,
    )


# --- SMA refinement (Pass 1 / Pass 2) ----------------------------------------


def sma_refinement_pass1_response_format() -> dict[str, Any]:
    return to_gateway_schema_from_dict(
        "sma_refinement_pass1",
        {
            "type": "object",
            "properties": {
                "field_statuses": {"type": "object"},
                "refined_corrections": {"type": "object"},
                "hitl_flags": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["field_statuses", "refined_corrections", "hitl_flags"],
            "additionalProperties": True,
        },
        strict=False,
    )


def sma_refinement_pass2_response_format() -> dict[str, Any]:
    return to_gateway_schema_from_dict(
        "sma_refinement_pass2",
        {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["items"],
            "additionalProperties": True,
        },
        strict=False,
    )


def response_format_for_ia_system_prompt(system: str) -> dict[str, Any] | None:
    if not genai_json_schema_enabled():
        return None
    if "You are IdentityAgent, responsible for inferring the grain contract" in system:
        return identity_grain_contract_response_format()
    if "You are IdentityAgent (term, **batch**)" in system:
        return identity_term_batch_envelope_response_format()
    if "code-generation assistant for IdentityAgent **grain**" in system:
        return identity_hook_spec_response_format()
    if "code-generation assistant for IdentityAgent **term**" in system:
        return identity_hook_spec_response_format()
    return None
