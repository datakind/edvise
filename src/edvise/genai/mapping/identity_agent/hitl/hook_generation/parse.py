"""Parse LLM text into :class:`HookSpec`."""

from __future__ import annotations

import json
import logging
from typing import Union

from edvise.genai.mapping.identity_agent.utilities import strip_json_fences
from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec

logger = logging.getLogger(__name__)

RawHookSpecInput = Union[str, bytes, dict]


def parse_hook_spec(raw: RawHookSpecInput) -> HookSpec:
    """
    Parse model output (optionally fenced) or a dict into :class:`HookSpec`.
    """
    if isinstance(raw, dict):
        return HookSpec.model_validate(raw)
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    text = strip_json_fences(text)
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("HookSpec parse failed to load JSON (truncated): %s", text[:500])
        raise
    return HookSpec.model_validate(d)


__all__ = ["parse_hook_spec"]
