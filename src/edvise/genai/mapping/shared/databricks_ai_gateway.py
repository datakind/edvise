"""
Databricks MLflow AI Gateway via the OpenAI-compatible client.

Shared by SchemaMappingAgent execution and IdentityAgent so execution code never imports
``identity_agent`` for gateway access only.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, Final, TypeVar, cast

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from edvise.genai.mapping.shared.utilities import (
    disable_mlflow_side_effects_for_openai_gateway,
)

MLFLOW_AI_GATEWAY_ON_WORKSPACE_PATH: Final[str] = "/ai-gateway/mlflow/v1"

DEFAULT_GATEWAY_MODEL_ID: str = "claude-sonnet-edvise-genai"

# System + user are concatenated into one role=user message (IA / SMA).
LLM_COMPLETE_SYSTEM_USER_SEP: Final[str] = "\n\n---\n\n"
DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS: Final[int] = 16_000

_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")


def llm_complete_combined_message_content(system: str, user: str) -> str:
    """Exact ``content`` string sent to the gateway for ``llm_complete(system, user)``."""
    return system + LLM_COMPLETE_SYSTEM_USER_SEP + user


def disable_mlflow_tracing_for_openai_gateway_client() -> None:
    """
    Turn off MLflow tracing / OpenAI autolog for gateway calls (see module docstring).

    Job scripts should also call
    :func:`~edvise.genai.mapping.shared.utilities.disable_mlflow_side_effects_for_openai_gateway`
    at import time **before** loading packages that import ``openai``.
    """
    disable_mlflow_side_effects_for_openai_gateway()


def _normalize_databricks_host(host: str) -> str:
    h = host.strip()
    for prefix in ("https://", "http://"):
        if h.lower().startswith(prefix):
            h = h[len(prefix) :]
            break
    return h.rstrip("/")


def _ai_gateway_cloud_segment_from_host(host: str | None) -> str:
    """
    Subdomain segment for ``{workspace_id}.ai-gateway.<segment>.databricks.com``.

    Override with ``AI_GATEWAY_CLOUD_SEGMENT`` when the host does not imply cloud.
    """
    explicit = (os.environ.get("AI_GATEWAY_CLOUD_SEGMENT") or "").strip()
    if explicit:
        return explicit
    if not host:
        return "gcp"
    h = host.lower()
    if ".gcp.databricks.com" in h:
        return "gcp"
    if ".azuredatabricks.net" in h:
        return "azure"
    return "cloud"


def build_mlflow_ai_gateway_base_url(
    *,
    workspace_id: str,
    cloud_segment: str,
) -> str:
    """MLflow OpenAI-compatible gateway URL for a Databricks workspace org id."""
    wid = workspace_id.strip()
    cloud = cloud_segment.strip()
    if not wid or not cloud:
        raise ValueError("workspace_id and cloud_segment are required")
    return f"https://{wid}.ai-gateway.{cloud}.databricks.com/mlflow/v1"


def resolve_databricks_workspace_host() -> str | None:
    """
    Workspace hostname for the current run (no scheme), if known.

    Uses ``DATABRICKS_HOST``, Databricks SDK default config, then notebook context.
    """
    raw = (os.environ.get("DATABRICKS_HOST") or "").strip()
    if raw:
        return _normalize_databricks_host(raw)

    try:
        from databricks.sdk.core import Config

        cfg_host = (Config().host or "").strip()
        if cfg_host:
            return _normalize_databricks_host(cfg_host)
    except Exception as exc:
        _LOG.debug("Databricks SDK host unavailable (%s)", exc)

    try:
        from databricks.sdk.runtime import dbutils

        # dbutils typing varies between local stubs and runtime; treat as Any here.
        dbutils_any = cast(Any, dbutils)
        ctx = dbutils_any.notebook.entry_point.getDbutils().notebook().getContext()
        for getter_name in ("browserHostName", "apiUrl"):
            try:
                getter = getattr(ctx, getter_name)
                val = getter()
                if val is not None and str(val).strip():
                    return _normalize_databricks_host(str(val))
            except Exception:
                continue
    except Exception as exc:
        _LOG.debug("dbutils workspace host unavailable (%s)", exc)

    return None


def resolve_databricks_workspace_id() -> str | None:
    """
    Numeric Databricks workspace / org id for the current job or notebook.

    Matches the id in AI Gateway URLs and ``?o=`` workspace query params.
    """
    for key in ("DATABRICKS_WORKSPACE_ID",):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return raw

    try:
        from databricks.sdk.runtime import dbutils

        dbutils_any = cast(Any, dbutils)
        wid = (
            dbutils_any.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .workspaceId()
            .get()
        )
        if wid is not None and str(wid).strip():
            return str(wid).strip()
    except Exception as exc:
        _LOG.debug("dbutils workspaceId unavailable (%s)", exc)

    return None


def resolve_ai_gateway_base_url() -> str:
    """Resolve the MLflow AI Gateway OpenAI base URL for the current workspace.

    Precedence: ``AI_GATEWAY_BASE_URL``; then workspace id + cloud from the job;
    then ``https://<workspace-host>/ai-gateway/mlflow/v1``. No hardcoded org default.
    """
    explicit = (os.environ.get("AI_GATEWAY_BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")

    host = resolve_databricks_workspace_host()
    workspace_id = resolve_databricks_workspace_id()

    if workspace_id:
        cloud = _ai_gateway_cloud_segment_from_host(host)
        url = build_mlflow_ai_gateway_base_url(
            workspace_id=workspace_id,
            cloud_segment=cloud,
        )
        _LOG.debug(
            "MLflow AI Gateway base URL from workspace_id=%s cloud=%s",
            workspace_id,
            cloud,
        )
        return url

    if host:
        url = f"https://{host}{MLFLOW_AI_GATEWAY_ON_WORKSPACE_PATH}"
        _LOG.debug("MLflow AI Gateway base URL from workspace host=%s", host)
        return url

    raise ValueError(
        "Cannot resolve MLflow AI Gateway base URL: run on Databricks compute (job or "
        "notebook) so workspace id/host are available, set DATABRICKS_HOST for local SDK "
        "auth, or set AI_GATEWAY_BASE_URL explicitly."
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


def is_retryable_openai_gateway_error(exc: BaseException) -> bool:
    """
    Whether to retry a failed OpenAI client call to the Databricks MLflow AI Gateway.

    Includes **403** because the gateway sometimes returns it for transient / policy blips;
    persistent ACL failures will exhaust :func:`wrap_llm_complete_with_retries` and still fail.
    """
    try:
        import openai
    except ImportError:
        return False
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        code = exc.status_code
        if code == 401:
            return False
        return code in (403, 408, 429, 500, 502, 503, 504)
    return False


def gateway_run_once_error_text_is_retryable(error_text: str) -> bool:
    """
    Best-effort match for :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.eval.run_once`
    failure strings (``HTTP 403: ...``) when exceptions are swallowed into a dict.
    """
    if not error_text:
        return False
    lower = error_text.lower()
    for code in ("403", "408", "429", "500", "502", "503", "504"):
        if f"http {code}" in lower:
            return True
    if any(
        s in lower
        for s in (
            "connection error",
            "connecttimeout",
            "read timed out",
            "timeout",
            "temporarily unavailable",
        )
    ):
        return True
    return False


def invoke_with_openai_retries(
    fn: Callable[[], _T],
    *,
    max_attempts: int = 5,
    initial_backoff_s: float = 2.0,
    max_backoff_s: float = 60.0,
    log: logging.Logger | None = None,
) -> _T:
    """Run ``fn`` until success or non-retryable failure / attempts exhausted."""
    log = log if log is not None else _LOG
    for attempt in range(max_attempts):
        try:
            return fn()
        except BaseException as exc:
            if (
                not is_retryable_openai_gateway_error(exc)
                or attempt >= max_attempts - 1
            ):
                raise
            delay = min(
                max_backoff_s,
                initial_backoff_s * (2**attempt),
            )
            delay *= 0.5 + random.random() * 0.5
            log.warning(
                "OpenAI gateway call failed (%s); retry %d/%d after %.1fs",
                type(exc).__name__,
                attempt + 1,
                max_attempts - 1,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError("invoke_with_openai_retries: unreachable")  # pragma: no cover


def wrap_llm_complete_with_retries(
    llm_complete: Callable[[str, str], str],
    *,
    max_attempts: int = 5,
    initial_backoff_s: float = 2.0,
    max_backoff_s: float = 60.0,
    log: logging.Logger | None = None,
) -> Callable[[str, str], str]:
    """Wrap ``llm_complete(system, user)`` with :func:`invoke_with_openai_retries` semantics."""
    log = log if log is not None else _LOG

    def wrapped(system: str, user: str) -> str:
        return invoke_with_openai_retries(
            lambda: llm_complete(system, user),
            max_attempts=max_attempts,
            initial_backoff_s=initial_backoff_s,
            max_backoff_s=max_backoff_s,
            log=log,
        )

    return wrapped
