"""Step 2b — transformation map schemas, utilities, prompts, and eval."""

from . import prompt_builder, schemas, utilities

__all__ = ["eval", "prompt_builder", "schemas", "utilities"]


def __getattr__(name: str):
    if name == "eval":
        from . import eval as eval_module
        return eval_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
