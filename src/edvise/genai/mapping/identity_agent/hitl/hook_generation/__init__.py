"""LLM hook generation for HITL ``generate_hook`` reentry (grain + term ``HookSpec``), plus module materialization."""

from edvise.genai.mapping.identity_agent.hitl.hook_generation.generate import (
    generate_hook_spec,
    generate_hook_specs_for_hook_items,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
    materialize_hook_spec_to_file,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    default_hook_module_relpath,
    ensure_hook_spec_file,
    resolve_hook_module_path,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.parse import (
    parse_hook_spec,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.prompt_builder import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.signature_check import (
    signature_mismatches,
)

__all__ = [
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "default_hook_module_relpath",
    "extract_config_snippet_for_hook_item",
    "generate_hook_spec",
    "generate_hook_specs_for_hook_items",
    "ensure_hook_spec_file",
    "materialize_hook_spec_to_file",
    "resolve_hook_module_path",
    "parse_hook_spec",
    "signature_mismatches",
]
