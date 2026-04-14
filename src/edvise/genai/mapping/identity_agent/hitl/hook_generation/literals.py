"""Coerce ``example_input`` / ``example_output`` strings for hook smoke tests and validate_hook."""

from __future__ import annotations

import ast
from typing import Any


def coerce_hook_example_value(value: Any) -> Any:
    """
    JSON stores examples as strings; parse Python literals with :func:`ast.literal_eval`.

    Non-string values (e.g. JSON numbers) are returned as-is.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


__all__ = ["coerce_hook_example_value"]
