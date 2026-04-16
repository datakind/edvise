"""Step 2a — field mapping manifest (sourcing spec, prompts, eval, helpers)."""

from __future__ import annotations

from typing import Any

__all__ = ["eval", "prompt_builder", "schemas", "sma_validation"]


def __getattr__(name: str) -> Any:
    if name == "schemas":
        from . import schemas as m

        return m
    if name == "prompt_builder":
        from . import prompt_builder as pb

        return pb
    if name == "eval":
        from . import eval as ev

        return ev
    if name == "sma_validation":
        from . import sma_validation as sv

        return sv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
