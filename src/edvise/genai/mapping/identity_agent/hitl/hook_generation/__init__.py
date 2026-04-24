"""LLM hook generation for HITL ``generate_hook`` reentry (grain + term ``HookSpec``), plus module materialization."""

from edvise.genai.mapping.identity_agent.hitl.hook_generation.generate import (
    generate_hook_spec,
    generate_hook_specs_for_hook_items,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
    materialize_hook_spec_to_file,
    materialize_hook_specs_to_file,
    merge_hook_specs,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    default_hook_module_relpath,
    ensure_hook_spec_file,
    hook_modules_root_from_bronze_volume,
    resolve_hook_module_path,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.parse import (
    parse_hook_spec,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.prompt import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
    normalized_column_names_from_raw_headers,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.signature_check import (
    signature_mismatches,
)

__all__ = [
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "default_hook_module_relpath",
    "extract_config_snippet_for_hook_item",
    "normalized_column_names_from_raw_headers",
    "generate_hook_spec",
    "generate_hook_specs_for_hook_items",
    "ensure_hook_spec_file",
    "hook_modules_root_from_bronze_volume",
    "materialize_hook_spec_to_file",
    "materialize_hook_specs_to_file",
    "merge_hook_specs",
    "resolve_hook_module_path",
    "parse_hook_spec",
    "signature_mismatches",
]
