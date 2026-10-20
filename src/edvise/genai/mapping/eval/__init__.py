"""Offline evaluation for GenAI mapping prompts.

Imports stay lazy. ``transformation.eval`` imports this package while it is
still initializing, and the Step 2b scorer imports that module back.
"""

from __future__ import annotations

from typing import Any

__all__ = ["assess_step2a", "assess_step2b", "resolve_eval_model", "run_once"]


def __getattr__(name: str) -> Any:
    if name == "run_once":
        from edvise.genai.mapping.eval.gateway import run_once

        return run_once
    if name == "resolve_eval_model":
        from edvise.genai.mapping.eval.models import resolve_eval_model

        return resolve_eval_model
    if name in {"assess_step2a", "assess_step2b"}:
        from edvise.genai.mapping.eval import scorers

        return getattr(scorers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
