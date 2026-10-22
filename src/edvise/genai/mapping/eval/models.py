"""Gateway model slots for Step 2a and Step 2b bake-offs.

Each slot defaults to Databricks' hosted Foundation Model API id
``system.ai.claude-<family>-<version>`` (dots in the version become dashes,
e.g. ``sonnet:4.5`` -> ``system.ai.claude-sonnet-4-5``). ``system.ai`` is a
Databricks-managed catalog, not the ``<catalog>.genai_mapping.*`` routes this
repo registers for production, so no workspace catalog is substituted in.
Override a slot explicitly, via ``EDVISE_EVAL_GATEWAY_MODELS`` JSON, or via a
per-slot environment variable if that naming doesn't match your workspace.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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


def default_system_ai_model_id(family: str, version: str) -> str:
    """Databricks-hosted Foundation Model API id for one slot.

    ``system.ai.claude-<family>-<version>``, e.g. ``sonnet:4.5`` ->
    ``system.ai.claude-sonnet-4-5``. Dots in the version become dashes
    because this is an unquoted Unity Catalog identifier.
    """
    return f"system.ai.claude-{family}-{version.replace('.', '-')}"


def resolve_comparison_models(
    models: str | list[str],
) -> list[ResolvedEvalModel]:
    """Resolve the models one bake-off compares.

    ``models`` is a comma-separated string, a JSON array, or a list of slots
    such as ``sonnet:4.5``. Every listed slot is scored in the same job,
    resolved through :func:`resolve_eval_model`'s usual precedence. At least
    two slots are required so the run compares them.
    """
    labels = _parse_model_list(models)
    if len(labels) < 2:
        raise ValueError(
            "A comparison needs at least two model slots. "
            f"Choose from: {', '.join(supported_model_labels())}."
        )
    return [resolve_eval_model(label) for label in labels]


def _parse_model_list(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        items = raw
    else:
        text = raw.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"models list must be JSON or comma-separated: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("models JSON must be an array of slot strings.")
            items = parsed
        else:
            items = text.split(",")
    labels: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"model slot {item!r} must be a string.")
        label = item.strip().lower()
        if not label:
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def resolve_eval_model(
    selection: str,
    *,
    gateway_model_id: str | None = None,
) -> ResolvedEvalModel:
    """Resolve one bake-off slot to a Databricks gateway model id.

    Precedence is the explicit id, then ``EDVISE_EVAL_GATEWAY_MODELS`` (a JSON
    object keyed by ``sonnet:4.5``), then the per-slot environment variable,
    then a non-empty registry default, then the hosted
    ``system.ai.claude-<family>-<version>`` foundation model.
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

    return ResolvedEvalModel(
        family, version, default_system_ai_model_id(family, version)
    )


def _parse_gateway_model_map(raw: str | dict[str, str]) -> dict[str, object]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"gateway model map must be a JSON object: {exc}") from exc
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        raise ValueError("gateway model map must be a JSON object.")
    return parsed


def _gateway_model_from_env_map(label: str) -> str | None:
    raw = os.environ.get(GATEWAY_MODELS_ENV, "").strip()
    if not raw:
        return None
    parsed = _parse_gateway_model_map(raw)
    value = parsed.get(label)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{GATEWAY_MODELS_ENV}[{label!r}] must be a gateway model id.")
    return value.strip()


# Notebook evaluators for Step 2a and Step 2b share this list. Mutate it in place
# (``MODELS[:] = [...]``) so both importers see the same object.
_DEFAULT_EVAL_MODELS: tuple[str, ...] = (
    "claude-opus-test-genai-ai-data-cleaning",
    "claude-sonnet-test-genai-ai-data-cleaning",
    "claude-haiku-test-genai-data-cleaning",
)

MODEL_BASE_SLUG = {
    "claude-opus-test-genai-ai-data-cleaning": "opus",
    "claude-sonnet-test-genai-ai-data-cleaning": "sonnet",
    "claude-haiku-test-genai-data-cleaning": "haiku",
}

MODELS: list[str] = list(_DEFAULT_EVAL_MODELS)

# Folder tag for Step 2b artifact directories, for example ``sonnet_2b_1shot``.
SHOT_TAG = "1shot"


def apply_eval_models_from_env() -> None:
    """If ``EDVISE_EVAL_MODELS`` is set, replace ``MODELS`` in place.

    Comma-separated entries; each entry is a short slug (``opus``, ``sonnet``, ``haiku``)
    or a full gateway model id from ``_DEFAULT_EVAL_MODELS``. Order is preserved;
    duplicates are dropped.
    """
    raw = os.environ.get("EDVISE_EVAL_MODELS", "").strip()
    if not raw:
        return
    slug_to_model = {MODEL_BASE_SLUG[model]: model for model in _DEFAULT_EVAL_MODELS}
    id_lower_to_model = {model.lower(): model for model in _DEFAULT_EVAL_MODELS}
    allowed = set(_DEFAULT_EVAL_MODELS)
    resolved: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        key = token.lower()
        if key in slug_to_model:
            model_id = slug_to_model[key]
        elif token in allowed:
            model_id = token
        elif key in id_lower_to_model:
            model_id = id_lower_to_model[key]
        else:
            slugs = ", ".join(sorted(slug_to_model))
            raise ValueError(
                f"EDVISE_EVAL_MODELS: unknown entry {token!r}. "
                f"Use comma-separated slugs ({slugs}) or a full gateway model id "
                "from the eval defaults."
            )
        if model_id not in seen:
            seen.add(model_id)
            resolved.append(model_id)
    if not resolved:
        raise ValueError("EDVISE_EVAL_MODELS produced an empty model list.")
    MODELS[:] = resolved
    logger.info("EDVISE_EVAL_MODELS active → %s", MODELS)
