"""Databricks MLflow AI Gateway via the OpenAI-compatible client (same stack as SMA eval)."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Final, cast

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.shared.mlflow_gateway_bootstrap import (
    disable_mlflow_side_effects_for_openai_gateway,
)

# Same default endpoint as ``schema_mapping_agent.manifest.eval`` (MLflow serving / gateway).
DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL: str = (
    "https://4437281602191762.ai-gateway.gcp.databricks.com/mlflow/v1"
)

DEFAULT_GATEWAY_MODEL_ID: str = "claude-sonnet-test-genai-ai-data-cleaning"

# System + user are concatenated into one role=user message (IA / SMA).
LLM_COMPLETE_SYSTEM_USER_SEP: Final[str] = "\n\n---\n\n"
DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS: Final[int] = 16_000

_LOG = logging.getLogger(__name__)


def llm_complete_combined_message_content(system: str, user: str) -> str:
    """Exact ``content`` string sent to the gateway for ``llm_complete(system, user)``."""
    return system + LLM_COMPLETE_SYSTEM_USER_SEP + user


def disable_mlflow_tracing_for_openai_gateway_client() -> None:
    """
    Turn off MLflow tracing / OpenAI autolog for gateway calls (see module docstring).

    Job scripts should also call :func:`~edvise.genai.mapping.shared.mlflow_gateway_bootstrap.disable_mlflow_side_effects_for_openai_gateway`
    at import time **before** loading packages that import ``openai``.
    """
    disable_mlflow_side_effects_for_openai_gateway()


def resolve_ai_gateway_base_url() -> str:
    """``AI_GATEWAY_BASE_URL`` env, else :data:`DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL`."""
    return os.environ.get(
        "AI_GATEWAY_BASE_URL", DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL
    )


def resolve_gateway_model_id() -> str:
    """``GATEWAY_MODEL_ID`` env, else :data:`DEFAULT_GATEWAY_MODEL_ID`."""
    return os.environ.get("GATEWAY_MODEL_ID", DEFAULT_GATEWAY_MODEL_ID)


def _token_from_authorization_header(headers: dict[str, str]) -> str | None:
    auth = headers.get("Authorization") or headers.get("authorization")
    if not auth or not isinstance(auth, str):
        return None
    parts = auth.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _token_from_databricks_sdk_default_auth() -> str | None:
    """
    On Databricks compute, ``Config().authenticate()`` often works when ``DATABRICKS_TOKEN``
    is unset (metadata service / OAuth for the job or notebook identity).
    """
    try:
        from databricks.sdk.core import Config
    except ImportError:
        _LOG.debug("databricks-sdk not installed; cannot resolve runtime workspace token")
        return None
    try:
        headers = Config().authenticate()
    except Exception as e:
        _LOG.debug("Databricks SDK default auth unavailable (%s)", e)
        return None
    return _token_from_authorization_header(headers)


def require_databricks_token() -> str:
    """
    Return a workspace token for the gateway ``api_key`` (PAT or OAuth bearer from the SDK).

    Order: ``DATABRICKS_TOKEN`` env, then :func:`_token_from_databricks_sdk_default_auth`
    (Databricks jobs / Repos when the env var is not injected).

    ``OPENAI_API_KEY`` is not used for this gateway.
    """
    token = (os.environ.get("DATABRICKS_TOKEN") or "").strip()
    if token:
        return token
    from_sdk = _token_from_databricks_sdk_default_auth()
    if from_sdk:
        return from_sdk
    msg = (
        "No Databricks workspace token for the MLflow AI gateway: set DATABRICKS_TOKEN "
        "(e.g. PAT or secret-backed env) or run on Databricks with databricks-sdk default "
        "credentials so Config().authenticate() succeeds. OPENAI_API_KEY is not used here."
    )
    raise ValueError(msg)


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
    disable_mlflow_tracing_for_openai_gateway_client()
    key = api_key if api_key is not None else require_databricks_token()
    url = base_url if base_url is not None else resolve_ai_gateway_base_url()
    return OpenAI(api_key=key, base_url=url)


def make_databricks_gateway_llm_complete(
    client: OpenAI,
    *,
    model: str | None = None,
    max_tokens: int = DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
) -> Callable[[str, str], str]:
    """
    Return ``llm_complete(system, user)`` for :mod:`~edvise.genai.mapping.identity_agent.grain_inference.runner`.

    The gateway is called with a single user message: ``system``, a separator, then ``user``
    (matches ``ia_dev`` / SMA notebook patterns).
    """
    resolved_model = model if model is not None else resolve_gateway_model_id()

    def complete(system: str, user: str) -> str:
        messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": "user", "content": llm_complete_combined_message_content(system, user)}],
        )
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return complete


def log_grain_hitl_queue(
    contract: GrainContract, *, logger: logging.Logger | None = None
) -> None:
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


def log_grain_auto_approve(
    contract: GrainContract, *, logger: logging.Logger | None = None
) -> None:
    """Log auto-approve path for one grain contract."""
    log = logger if logger is not None else _LOG
    log.info("Auto-approve: %s confidence= %s", contract.table, contract.confidence)
