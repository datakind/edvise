"""Parse LLM text into :class:`~edvise.genai.mapping.shared.hitl.hook_spec.schemas.HookSpec`."""

from __future__ import annotations

import json
import logging
from typing import Union

from .schemas import HookSpec
from edvise.genai.mapping.shared.utilities import strip_json_fences

logger = logging.getLogger(__name__)

RawHookSpecInput = Union[str, bytes, dict]


def parse_hook_spec(raw: RawHookSpecInput) -> HookSpec:
    """
    Parse model output (optionally fenced) or a dict into :class:`HookSpec`.

    ``file`` may be omitted; :func:`~edvise.genai.mapping.shared.hitl.hook_spec.paths.ensure_hook_spec_file`
    assigns the canonical path before persisting.
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


__all__ = ["RawHookSpecInput", "parse_hook_spec"]
