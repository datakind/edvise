"""Step 2b — transformation map schemas, utilities, prompts, and eval."""

from __future__ import annotations

from typing import Any

from . import prompt, schemas, utilities

__all__ = ["eval", "prompt", "schemas", "utilities"]


def __getattr__(name: str) -> Any:
    """Load scoring on demand so production imports do not initialize it."""
    if name == "eval":
        from . import eval as evaluation

        return evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
