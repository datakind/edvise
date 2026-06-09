"""LLM hook generation for HITL ``generate_hook`` reentry (grain + term ``HookSpec``), preview JSON, and materialization."""

from edvise.genai.mapping.identity_agent.hitl.hook_generation.generate import (
    generate_hook_spec,
    generate_hook_specs_for_hook_items,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
    materialize_hook_spec_to_file,
    materialize_hook_specs_to_file,
    merge_hook_specs,
)
from edvise.genai.mapping.shared.hitl.hook_spec.parse import parse_hook_spec
from edvise.genai.mapping.shared.hitl.hook_spec.paths import (
    default_hook_module_relpath,
    ensure_hook_spec_file,
    hook_modules_root_from_bronze_volume,
    resolve_hook_module_path,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.prompt import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
    normalized_column_names_from_raw_headers,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.preview import (
    apply_term_hook_preview_names_from_item_id,
    apply_term_hook_spec_names_from_item_id,
    assemble_hook_spec_drafts_as_module_text,
    hook_slug_from_item_id,
    write_identity_hook_preview_json,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.signature_check import (
    signature_mismatches,
)

__all__ = [
    "apply_term_hook_preview_names_from_item_id",
    "apply_term_hook_spec_names_from_item_id",
    "assemble_hook_spec_drafts_as_module_text",
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "default_hook_module_relpath",
    "extract_config_snippet_for_hook_item",
    "hook_slug_from_item_id",
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
    "write_identity_hook_preview_json",
]
