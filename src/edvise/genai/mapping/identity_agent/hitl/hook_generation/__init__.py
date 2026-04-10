"""LLM hook generation for HITL ``generate_hook`` reentry (grain + term ``HookSpec``)."""

from edvise.genai.mapping.identity_agent.hitl.hook_generation.generate import (
    generate_hook_spec,
    generate_hook_specs_for_hook_items,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.parse import (
    parse_hook_spec,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.prompt_builder import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
)

__all__ = [
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "extract_config_snippet_for_hook_item",
    "generate_hook_spec",
    "generate_hook_specs_for_hook_items",
    "parse_hook_spec",
]
