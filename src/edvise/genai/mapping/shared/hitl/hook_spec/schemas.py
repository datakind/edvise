"""Shared Pydantic models for generated hook specs and HITL domain routing (IA + SMA)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HookFunctionSpec(BaseModel):
    """
    Spec for one generated hook function.

    ``draft`` is the **full function definition** (``def`` line through body) as one string.
    ``signature`` is optional human/LLM metadata; materialize does not use it.
    For **term** hooks, prompts require ``example_input`` / ``example_output`` to be
    literal-evaluable for smoke tests. For **grain** (and similar domains), they are
    free-form documentation strings only.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    signature: str | None = Field(
        default=None,
        description="Optional; not used by materialize (draft carries the full def).",
    )
    description: str
    example_input: str | None = None
    example_output: str | int | float | None = None
    draft: str | None = Field(
        default=None,
        description="Complete Python function definition: signature and body as one string.",
    )


class HookSpec(BaseModel):
    """
    File path + function specs for a generated hook.
    Shared shape between term extraction hooks, dedup hooks, and SMA transform hooks.

    ``file`` is optional in raw LLM JSON; hook generation overwrites it with a canonical path
    from ``institution_id`` and domain before persisting to config.
    """

    model_config = ConfigDict(extra="forbid")

    file: str | None = Field(
        default=None,
        description=(
            "Relative path to the materialized module; set by the pipeline, not the LLM "
            "(e.g. identity_hooks/<institution_id>/dedup_hooks.py under bronze_volumes_path)."
        ),
    )
    functions: list[HookFunctionSpec]


class HITLDomain(str, Enum):
    IDENTITY_GRAIN = "identity_grain"
    IDENTITY_TERM = "identity_term"
    SMA_GRAIN = "sma_grain"
    SCHEMA_MAPPING = "schema_mapping"  # future
    TRANSFORM = "transform"  # future


__all__ = ["HITLDomain", "HookFunctionSpec", "HookSpec"]
