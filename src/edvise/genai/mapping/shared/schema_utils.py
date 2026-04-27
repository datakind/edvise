"""Convert Pydantic models (or explicit JSON Schema dicts) for OpenAI-style LLM gateways."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


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
