"""
Databricks Foundation Model API JSON-schema response_format helpers.

The MLflow / OpenAI-compatible gateway used by edvise supports::

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": str, "strict": bool, "schema": <JSON Schema>},
    }

We keep schemas intentionally small: top-level shape + a few required keys, with
``additionalProperties: true`` for nested objects so we do not fight Databricks
JSON Schema subset limitations (``anyOf`` / ``$ref``, etc.).

Opt-out: set env ``EDVISE_GENAI_JSON_SCHEMA=0`` to disable and fall back to plain
JSON prompting + ``strip_json_fences`` parsing.
"""

from __future__ import annotations

import os
from typing import Any

_EDVISE_GENAI_JSON_SCHEMA_ENV = "EDVISE_GENAI_JSON_SCHEMA"


def genai_json_schema_enabled() -> bool:
    v = (os.environ.get(_EDVISE_GENAI_JSON_SCHEMA_ENV) or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _wrap(name: str, schema: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": strict, "schema": schema},
    }


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
    return _wrap(
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
    return _wrap(
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
    return _wrap(
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


def identity_grain_contract_response_format() -> dict[str, Any]:
    return _wrap(
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
                "dedup_policy": {"type": "object"},
                "row_selection_required": {"type": "boolean"},
                "join_keys_for_2a": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {"type": "number"},
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
            "additionalProperties": True,
        },
        strict=False,
    )


# --- IdentityAgent: term batch (InstitutionTermContract) ----------------------


def identity_term_batch_envelope_response_format() -> dict[str, Any]:
    return _wrap(
        "identity_term_batch_envelope",
        {
            "type": "object",
            "properties": {
                "institution_id": {"type": "string"},
                "datasets": {"type": "object"},
            },
            "required": ["institution_id", "datasets"],
            "additionalProperties": True,
        },
        strict=False,
    )


# --- IdentityAgent: hook generation (HookSpec) -----------------------------


def identity_hook_spec_response_format() -> dict[str, Any]:
    return _wrap(
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
    return _wrap(
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
    return _wrap(
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
