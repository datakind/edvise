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

    * ``json.JSONDecodeError`` from ``parse_fn``: retry with the same
      ``call_fn`` contract (``correction_hint`` is set to ``None``).
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
