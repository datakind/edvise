"""
OpenAI / Databricks MLflow-style ``response_format`` helpers for edvise.

**Core**
    :func:`to_gateway_schema_from_dict` and :func:`to_gateway_schema` build the envelope::

        { "type": "json_schema", "json_schema": { "name", "strict", "schema" } }

    (inlines ``$defs`` / ``$ref`` when present).

**Named response shapes**
    Identity and Schema Mapping Agent response bodies are **derived** from Pydantic in
    :mod:`edvise.genai.mapping.shared.gateway_schema_derive` (``to_gateway_schema`` + ref inlining
    + grain ``if``/``then`` for HITL). SMA uses loose list/dict field types where the LLM
    output is normalized later in the pipeline.

    **Default: off.** Opt-in: set env ``EDVISE_GENAI_JSON_SCHEMA=1`` to send gateway
    ``response_format`` JSON schema; otherwise plain JSON prompting +
    ``strip_json_fences`` parsing.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Core: build response_format from JSON Schema or Pydantic
# ---------------------------------------------------------------------------


def to_gateway_schema_from_dict(
    name: str,
    schema: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Build ``response_format`` payload from an explicit JSON Schema object.

    Use for hand-tuned JSON Schema dicts; :func:`to_gateway_schema` delegates here.
    The input ``schema`` dict is not mutated.
    """
    inlined = _inline_refs(copy.deepcopy(schema))
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": inlined,
        },
    }


def to_gateway_schema(model: type[BaseModel], name: str, strict: bool = False) -> dict:
    return to_gateway_schema_from_dict(name, model.model_json_schema(), strict=strict)


def inline_json_schema_refs(model_json: dict[str, Any]) -> dict[str, Any]:
    """
    Fully inline ``$ref`` / ``$defs`` from a :meth:`pydantic.BaseModel.model_json_schema` dict.

    Databricks-style gateways often reject ``$ref``; use this before ``to_gateway_schema_from_dict``.
    """
    return _inline_refs(copy.deepcopy(model_json))


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $defs/$ref inline so gateway validators don't choke."""
    defs = schema.pop("$defs", {})
    return _resolve(copy.deepcopy(schema), defs)


def _resolve(obj: Any, defs: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref_name = obj["$ref"].split("/")[-1]
            return _resolve(copy.deepcopy(defs[ref_name]), defs)
        return {k: _resolve(v, defs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(i, defs) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Env: optional JSON schema mode
# ---------------------------------------------------------------------------

_EDVISE_GENAI_JSON_SCHEMA_ENV = "EDVISE_GENAI_JSON_SCHEMA"


def genai_json_schema_enabled() -> bool:
    v = (os.environ.get(_EDVISE_GENAI_JSON_SCHEMA_ENV) or "0").strip().lower()
    return v not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Schema Mapping Agent — Pydantic-derived in gateway_schema_derive
# ---------------------------------------------------------------------------


def mapping_manifest_envelope_response_format() -> dict[str, Any]:
    """
    Top-level mapping manifest JSON written by Step 2a (envelope; pipeline may inject ids).
    """
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_sma_mapping_manifest_envelope_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "mapping_manifest_envelope",
        build_sma_mapping_manifest_envelope_gateway_json_schema(),
        strict=False,
    )


def step2a_entity_pass_response_format() -> dict[str, Any]:
    """
    One Step 2a per-entity gateway call (cohort pass or course pass).

    ``merge_step2a_entity_manifests`` accepts either a top-level ``manifests`` fragment
    with a single entity key, *or* a :class:`FieldMappingManifest` top-level object, so
    the JSON schema only enforces a JSON object.
    """
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_sma_step2a_entity_pass_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "step2a_entity_pass",
        build_sma_step2a_entity_pass_gateway_json_schema(),
        strict=False,
    )


def transformation_map_wrapper_response_format() -> dict[str, Any]:
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_sma_transformation_map_wrapper_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "transformation_map_wrapper",
        build_sma_transformation_map_wrapper_gateway_json_schema(),
        strict=False,
    )


# ---------------------------------------------------------------------------
# IdentityAgent: grain, term batch, hook — Pydantic-derived in gateway_schema_derive
# ---------------------------------------------------------------------------


def identity_grain_contract_response_format() -> dict[str, Any]:
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_identity_grain_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "identity_grain_contract",
        build_identity_grain_gateway_json_schema(),
        strict=False,
    )


def identity_term_batch_envelope_response_format() -> dict[str, Any]:
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_identity_term_batch_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "identity_term_batch_envelope",
        build_identity_term_batch_gateway_json_schema(),
        strict=False,
    )


def identity_hook_spec_response_format() -> dict[str, Any]:
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_identity_hook_spec_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "identity_hook_spec",
        build_identity_hook_spec_gateway_json_schema(),
        strict=False,
    )


# ---------------------------------------------------------------------------
# SMA refinement (Pass 1 / Pass 2) — Pydantic-derived in gateway_schema_derive
# ---------------------------------------------------------------------------


def sma_refinement_pass1_response_format() -> dict[str, Any]:
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_sma_refinement_pass1_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "sma_refinement_pass1",
        build_sma_refinement_pass1_gateway_json_schema(),
        strict=False,
    )


def sma_refinement_pass2_response_format() -> dict[str, Any]:
    from edvise.genai.mapping.shared.gateway_schema_derive import (
        build_sma_refinement_pass2_gateway_json_schema,
    )

    return to_gateway_schema_from_dict(
        "sma_refinement_pass2",
        build_sma_refinement_pass2_gateway_json_schema(),
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
