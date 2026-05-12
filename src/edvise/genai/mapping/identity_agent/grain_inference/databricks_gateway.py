"""Databricks MLflow AI Gateway via the OpenAI-compatible client (same stack as SMA eval)."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.shared import databricks_ai_gateway as _databricks_ai_gateway

DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL = (
    _databricks_ai_gateway.DEFAULT_DATABRICKS_MLFLOW_AI_GATEWAY_URL
)
DEFAULT_GATEWAY_MODEL_ID = _databricks_ai_gateway.DEFAULT_GATEWAY_MODEL_ID
LLM_COMPLETE_SYSTEM_USER_SEP = _databricks_ai_gateway.LLM_COMPLETE_SYSTEM_USER_SEP
DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS = (
    _databricks_ai_gateway.DEFAULT_GATEWAY_COMPLETION_MAX_TOKENS
)
llm_complete_combined_message_content = (
    _databricks_ai_gateway.llm_complete_combined_message_content
)
disable_mlflow_tracing_for_openai_gateway_client = (
    _databricks_ai_gateway.disable_mlflow_tracing_for_openai_gateway_client
)
resolve_ai_gateway_base_url = _databricks_ai_gateway.resolve_ai_gateway_base_url
resolve_gateway_model_id = _databricks_ai_gateway.resolve_gateway_model_id
require_databricks_token = _databricks_ai_gateway.require_databricks_token
create_openai_client_for_databricks_gateway = (
    _databricks_ai_gateway.create_openai_client_for_databricks_gateway
)
make_databricks_gateway_llm_complete = (
    _databricks_ai_gateway.make_databricks_gateway_llm_complete
)

_LOG = logging.getLogger(__name__)

_T = TypeVar("_T")


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
