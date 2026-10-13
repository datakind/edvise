"""Gateway model slots for Step 2a bake-offs.

Endpoint ids are workspace-specific, so the six Sonnet and Haiku slots are
registered without hardcoded ids. Supply an id per run or through the
environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

SUPPORTED_STEP2A_MODELS: dict[tuple[str, str], str | None] = {
    ("sonnet", "4.5"): None,
    ("sonnet", "5"): None,
    ("sonnet", "5.5"): None,
    ("haiku", "4.5"): None,
    ("haiku", "5"): None,
    ("haiku", "5.5"): None,
}

GATEWAY_MODELS_ENV = "EDVISE_EVAL_GATEWAY_MODELS"


@dataclass(frozen=True)
class ResolvedEvalModel:
    family: str
    version: str
    gateway_model_id: str

    @property
    def selection(self) -> str:
        return f"{self.family}:{self.version}"


def supported_model_labels() -> list[str]:
    return [f"{family}:{version}" for family, version in SUPPORTED_STEP2A_MODELS]


def parse_model_selection(selection: str) -> tuple[str, str]:
    """Parse ``sonnet:4.5`` into a supported family and version."""
    token = selection.strip().lower()
    if ":" not in token:
        raise ValueError(
            f"Model selection {selection!r} must look like 'sonnet:4.5'. "
            f"Choose one of: {', '.join(supported_model_labels())}."
        )
    family, version = token.split(":", 1)
    if (family, version) not in SUPPORTED_STEP2A_MODELS:
        raise ValueError(
            f"Unsupported model {selection!r}. "
            f"Choose one of: {', '.join(supported_model_labels())}."
        )
    return family, version


def model_slot_env_var(family: str, version: str) -> str:
    """Environment variable that overrides one slot, e.g. ``EDVISE_EVAL_MODEL_SONNET_4_5``."""
    return f"EDVISE_EVAL_MODEL_{family.upper()}_{version.replace('.', '_')}"


def resolve_eval_model(
    selection: str,
    *,
    gateway_model_id: str | None = None,
) -> ResolvedEvalModel:
    """Resolve one bake-off slot to a Databricks gateway model id.

    Precedence is the explicit id, then ``EDVISE_EVAL_GATEWAY_MODELS`` (a JSON
    object keyed by ``sonnet:4.5``), then the per-slot environment variable,
    then a non-empty registry default.
    """
    family, version = parse_model_selection(selection)
    label = f"{family}:{version}"
    explicit = (gateway_model_id or "").strip()
    if explicit:
        return ResolvedEvalModel(family, version, explicit)

    from_map = _gateway_model_from_env_map(label)
    if from_map:
        return ResolvedEvalModel(family, version, from_map)

    from_slot = os.environ.get(model_slot_env_var(family, version), "").strip()
    if from_slot:
        return ResolvedEvalModel(family, version, from_slot)

    seeded = SUPPORTED_STEP2A_MODELS[(family, version)]
    if seeded and seeded.strip():
        return ResolvedEvalModel(family, version, seeded.strip())

    raise ValueError(
        f"No gateway model id configured for {label}. Pass --gateway_model_id, "
        f"set {GATEWAY_MODELS_ENV} to a JSON object, or set "
        f"{model_slot_env_var(family, version)}."
    )


def _gateway_model_from_env_map(label: str) -> str | None:
    raw = os.environ.get(GATEWAY_MODELS_ENV, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{GATEWAY_MODELS_ENV} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{GATEWAY_MODELS_ENV} must be a JSON object.")
    value = parsed.get(label)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{GATEWAY_MODELS_ENV}[{label!r}] must be a gateway model id.")
    return value.strip()
