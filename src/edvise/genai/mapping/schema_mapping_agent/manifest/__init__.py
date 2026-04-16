"""Step 2a — field mapping manifest (sourcing spec, prompts, eval, helpers)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["eval", "hitl", "hitl_resolver", "prompt_builder", "schemas", "validation"]


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
    if name == "validation":
        from . import validation as v

        return v
    if name == "hitl":
        # SMA HITL lives at package level (``schema_mapping_agent.hitl``), not under manifest.
        return importlib.import_module("edvise.genai.mapping.schema_mapping_agent.hitl")
    if name == "hitl_resolver":
        return importlib.import_module("edvise.genai.mapping.schema_mapping_agent.hitl.resolver")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
