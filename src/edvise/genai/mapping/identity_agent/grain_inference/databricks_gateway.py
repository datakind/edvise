"""Databricks MLflow AI Gateway via the OpenAI-compatible client (same stack as SMA eval)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from openai import OpenAI

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract

# Same default endpoint as ``schema_mapping_agent.manifest.eval`` (MLflow serving / gateway).
DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL: str = (
    "https://4437281602191762.ai-gateway.gcp.databricks.com/mlflow/v1"
)

DEFAULT_GATEWAY_MODEL_ID: str = "claude-sonnet-test-genai-ai-data-cleaning"

_LOG = logging.getLogger(__name__)


def resolve_ai_gateway_base_url() -> str:
    """``AI_GATEWAY_BASE_URL`` env, else :data:`DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL`."""
    return os.environ.get("AI_GATEWAY_BASE_URL", DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL)


def resolve_gateway_model_id() -> str:
    """``GATEWAY_MODEL_ID`` env, else :data:`DEFAULT_GATEWAY_MODEL_ID`."""
    return os.environ.get("GATEWAY_MODEL_ID", DEFAULT_GATEWAY_MODEL_ID)


def require_databricks_token() -> str:
    """
    Return ``DATABRICKS_TOKEN`` for the gateway ``api_key``.

    Raises ``ValueError`` if unset (``OPENAI_API_KEY`` is not used for this gateway).
    """
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        msg = (
            "DATABRICKS_TOKEN is required for the Databricks MLflow AI gateway (same as SMA). "
            "Bare OpenAI() uses OPENAI_API_KEY and is the wrong token for this gateway."
        )
        raise ValueError(msg)
    return token


def create_openai_client_for_databricks_gateway(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """
    Build an :class:`openai.OpenAI` client pointed at the Databricks gateway.

    If ``api_key`` is omitted, :func:`require_databricks_token` is used.
    If ``base_url`` is omitted, :func:`resolve_ai_gateway_base_url` is used.
    """
    key = api_key if api_key is not None else require_databricks_token()
    url = base_url if base_url is not None else resolve_ai_gateway_base_url()
    return OpenAI(api_key=key, base_url=url)


def make_databricks_gateway_llm_complete(
    client: OpenAI,
    *,
    model: str | None = None,
    max_tokens: int = 16_000,
) -> Callable[[str, str], str]:
    """
    Return ``llm_complete(system, user)`` for :mod:`~edvise.genai.mapping.identity_agent.grain_inference.runner`.

    The gateway is called with a single user message: ``system``, a separator, then ``user``
    (matches ``ia_dev`` / SMA notebook patterns).
    """
    resolved_model = model if model is not None else resolve_gateway_model_id()

    def complete(system: str, user: str) -> str:
        messages = [{"role": "user", "content": system + "\n\n---\n\n" + user}]
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return complete


def log_grain_hitl_queue(contract: GrainContract, *, logger: logging.Logger | None = None) -> None:
    """Log HITL routing for one grain contract (structured ``hitl_items`` or legacy ``hitl_question``)."""
    log = logger if logger is not None else _LOG
    items = getattr(contract, "hitl_items", None)
    if items:
        for it in items:
            if isinstance(it, dict):
                q = it.get("hitl_question", it)
            else:
                q = getattr(it, "hitl_question", str(it))
            log.info(
                "HITL queue: %s confidence= %s | %s",
                contract.table,
                contract.confidence,
                q,
            )
    else:
        log.info(
            "HITL queue: %s confidence= %s | %s",
            contract.table,
            contract.confidence,
            getattr(contract, "hitl_question", None),
        )


def log_grain_auto_approve(contract: GrainContract, *, logger: logging.Logger | None = None) -> None:
    """Log auto-approve path for one grain contract."""
    log = logger if logger is not None else _LOG
    log.info("Auto-approve: %s confidence= %s", contract.table, contract.confidence)
