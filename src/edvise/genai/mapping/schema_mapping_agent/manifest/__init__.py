"""Step 2a — field mapping manifest (sourcing spec, prompts, eval, helpers)."""

from __future__ import annotations

__all__ = ["eval", "prompt_builder", "schemas"]


def __getattr__(name: str):
    if name == "schemas":
        from . import schemas as m

        return m
    if name == "prompt_builder":
        from . import prompt_builder as pb

        return pb
    if name == "eval":
        from . import eval as ev

        return ev
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
