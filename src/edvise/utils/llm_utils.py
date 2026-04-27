"""Utilities for LLM string responses, parsing, and retry policy."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import ValidationError

T = TypeVar("T")


class LLMRetryExhausted(Exception):
    """Raised when an LLM call and parse pipeline exhausts all retry attempts."""

    def __init__(self, last_error: Exception, last_raw_response: str) -> None:
        self.last_error = last_error
        self.last_raw_response = last_raw_response
        super().__init__(f"LLM retries exhausted: {last_error!r}")


def call_with_retry(
    call_fn: Callable[[str | None], str],
    parse_fn: Callable[[str], T],
    max_retries: int = 3,
    logger: logging.Logger | None = None,
) -> T:
    """Call ``call_fn`` with retries for malformed JSON and validation failures.

    * ``json.JSONDecodeError`` from ``parse_fn``: retry, usually with the same
      request (``correction_hint`` is ``None``). If the model returned
      *nothing* (empty or whitespace-only text), the next call includes an
      explicit instruction to emit valid JSON, since a blind repeat is futile.
    * :class:`pydantic.ValidationError` from ``parse_fn``: pass a
      ``correction_hint`` describing the prior raw output and the error
      on the next ``call_fn`` call.

    At most ``max_retries`` total invocations of ``call_fn`` are made.
    """
    if max_retries < 1:
        msg = "max_retries must be at least 1"
        raise ValueError(msg)

    correction_hint: str | None = None
    last_raw = ""

    for attempt in range(1, max_retries + 1):
        last_raw = call_fn(correction_hint)
        try:
            return parse_fn(last_raw)
        except json.JSONDecodeError as e:
            if logger is not None:
                logger.warning(
                    "LLM call attempt %s failed: %s: %s",
                    attempt,
                    type(e).__name__,
                    e,
                )
            if attempt == max_retries:
                raise LLMRetryExhausted(e, last_raw) from e
            if not (last_raw or "").strip():
                correction_hint = (
                    "Your previous response was empty (no text). "
                    "You must return exactly one JSON object and nothing else "
                    "before or after (no markdown fences unless asked)."
                )
            else:
                correction_hint = None
        except ValidationError as e:
            if logger is not None:
                logger.warning(
                    "LLM call attempt %s failed: %s: %s",
                    attempt,
                    type(e).__name__,
                    e,
                )
            if attempt == max_retries:
                raise LLMRetryExhausted(e, last_raw) from e
            correction_hint = (
                f"Your previous response was:\n\n{last_raw}\n\n"
                f"It failed validation with this error:\n\n{e}\n\n"
                "Return a corrected version."
            )
    assert False, "unreachable"  # noqa: S101


def llm_complete_with_parse_retry(
    llm_complete: Callable[[str, str], str],
    system: str,
    user: str,
    parse_fn: Callable[[str], T],
    max_retries: int = 3,
    logger: logging.Logger | None = None,
) -> T:
    """Call ``llm_complete(system, user)`` and run ``parse_fn`` on the returned text.

    Retries use :func:`call_with_retry`. On a :class:`pydantic.ValidationError`, the
    next attempt appends the correction block to the **user** message only; the
    system prompt is unchanged.
    """
    def call_fn(hint: str | None) -> str:
        if hint is None:
            return llm_complete(system, user)
        return llm_complete(system, f"{user}\n\n{hint}")

    return call_with_retry(call_fn, parse_fn, max_retries=max_retries, logger=logger)
