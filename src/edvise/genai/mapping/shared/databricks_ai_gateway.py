"""
Databricks MLflow AI Gateway via the OpenAI-compatible client.

Shared by SchemaMappingAgent execution and IdentityAgent so execution code never imports
``identity_agent`` for gateway access only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Final, cast

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from edvise.genai.mapping.shared.mlflow_gateway_bootstrap import (
    disable_mlflow_side_effects_for_openai_gateway,
)

# Same default endpoint as ``schema_mapping_agent.manifest.eval`` (MLflow serving / gateway).
DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL: str = (
    "https://4437281602191762.ai-gateway.gcp.databricks.com/mlflow/v1"
)

DEFAULT_GATEWAY_MODEL_ID: str = "claude-sonnet-edvise-genai"

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

    Job scripts should also call
    :func:`~edvise.genai.mapping.shared.mlflow_gateway_bootstrap.disable_mlflow_side_effects_for_openai_gateway`
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
    Resolve a short-lived workspace bearer via ``Config().authenticate()`` (Databricks SDK).

    Typical sources: job/cluster identity metadata service, OAuth M2M / service principal,
    or a local ``databricks auth login`` profile when ``DATABRICKS_HOST`` is set.
    """
    try:
        from databricks.sdk.core import Config
    except ImportError:
        _LOG.debug(
            "databricks-sdk not installed; cannot resolve runtime workspace token"
        )
        return None
    try:
        headers = Config().authenticate()
    except Exception as e:
        _LOG.debug("Databricks SDK default auth unavailable (%s)", e)
        return None
    return _token_from_authorization_header(headers)


def require_databricks_token() -> str:
    """
    Return a workspace bearer for the gateway ``api_key`` via
    :func:`_token_from_databricks_sdk_default_auth`.

    Personal access tokens (``DATABRICKS_TOKEN``) are not used for this path.

    ``OPENAI_API_KEY`` is not used for this gateway.
    """
    from_sdk = _token_from_databricks_sdk_default_auth()
    if from_sdk:
        return from_sdk
    msg = (
        "No Databricks workspace token for the MLflow AI gateway: databricks-sdk "
        "Config().authenticate() did not return a Bearer token. Run on Databricks compute "
        "with job/cluster identity, configure OAuth / service principal credentials, or "
        "use ``databricks auth login`` locally with DATABRICKS_HOST set. "
        "OPENAI_API_KEY is not used here."
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


def _text_from_message_content(
    content: object,
) -> str:
    """
    Best-effort string from ``message.content`` (OpenAI is usually ``str | None``;
    some routes may return list-shaped multimodal content).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block.get("text", "")))
                else:
                    t = block.get("text")
                    if t is not None:
                        parts.append(str(t))
            else:
                tx = getattr(block, "text", None)
                if tx is not None:
                    parts.append(str(tx))
        return "".join(parts)
    return str(content)


def _assistant_text_from_chat_completion_or_raise(
    resp: object, *, log: logging.Logger, default_model: str | None = None
) -> str:
    """
    Return the assistant's output text, or raise if there is nothing usable to parse as JSON.

    A ``200`` response with ``content=None`` and no text was previously turned into ``""``,
    which only surfaces as JSONDecodeError on empty input. We fail fast with diagnostics
    and surface refusals (e.g. Claude) explicitly.
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        msg = "AI Gateway returned no choices on chat.completions"
        log.error("%s: model=%r", msg, getattr(resp, "model", default_model))
        raise RuntimeError(msg) from None

    ch0 = choices[0]
    msg = ch0.message
    raw = _text_from_message_content(getattr(msg, "content", None))
    if raw.strip():
        return raw

    ref = getattr(msg, "refusal", None)
    if isinstance(ref, str) and ref.strip():
        short = ref.strip()[:2000]
        log.error(
            "AI Gateway: model refusal (not valid JSON for downstream parse): %s", short
        )
        raise RuntimeError(
            "The model refused to return structured output. Refusal: "
            + ref.strip()[:4000]
        ) from None

    u = getattr(resp, "usage", None)
    udump: object
    if u is not None and hasattr(u, "model_dump"):
        udump = u.model_dump()  # type: ignore[assignment]
    else:
        udump = u
    fr = getattr(ch0, "finish_reason", None)
    mod = getattr(resp, "model", None) or default_model
    c_raw = getattr(msg, "content", None)
    log.error(
        "AI Gateway: empty assistant message: finish_reason=%r model=%r usage=%r content=%r",
        fr,
        mod,
        udump,
        c_raw,
    )
    raise RuntimeError(
        "AI Gateway returned an empty assistant message. "
        f"finish_reason={fr!r}, model={mod!r}, usage={udump!r}. "
        "The prompt may exceed the model context, max_tokens may be exhausted, "
        "or the model emitted no text — try a smaller input batch or higher limits."
    ) from None


def make_databricks_gateway_llm_complete(
    client: OpenAI,
    *,
    model: str | None = None,
    max_tokens: int = DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS,
) -> Callable[[str, str], str]:
    """
    Return ``llm_complete(system, user)``.

    The gateway is called with a single user message: ``system``, a separator, then ``user``
    (matches ``ia_dev`` / SMA notebook patterns).
    """
    resolved_model = model if model is not None else resolve_gateway_model_id()

    def complete(system: str, user: str) -> str:
        messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": "user",
                    "content": llm_complete_combined_message_content(system, user),
                }
            ],
        )
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return _assistant_text_from_chat_completion_or_raise(
            resp, log=_LOG, default_model=resolved_model
        )

    return complete
