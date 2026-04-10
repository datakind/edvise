"""
HITL Pydantic model source for IdentityAgent system prompts (SMA-style ``inspect.getsource``).
"""

from __future__ import annotations

from edvise.genai.mapping.identity_agent.utilities import concat_model_sources


def get_hitl_item_schema_context() -> str:
    """Python source for ``HITLItem`` and nested option / resolution types (grain + term)."""
    from edvise.genai.mapping.identity_agent.hitl.schemas import (
        GrainResolution,
        HITLDomain,
        HITLItem,
        HITLOption,
        HITLResolution,
        HITLStatus,
        HITLTarget,
        ReentryDepth,
        TermResolution,
    )

    return concat_model_sources(
        (
            HITLDomain,
            ReentryDepth,
            HITLStatus,
            GrainResolution,
            TermResolution,
            HITLOption,
            HITLTarget,
            HITLResolution,
            HITLItem,
        )
    )


__all__ = ["get_hitl_item_schema_context"]
