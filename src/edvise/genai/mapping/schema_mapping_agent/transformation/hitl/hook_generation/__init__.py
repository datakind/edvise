"""LLM HookSpec generation for Step 2b ``hook_required`` plans, preview JSON, and helpers (mirrors IA ``hitl/hook_generation``)."""

from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation.generate import (
    generate_sma_transform_hook_preview_rows_for_entity,
    generate_sma_transform_hook_spec,
    load_hook_specs_from_sma_preview_path,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation.prompt import (
    build_sma_transform_hook_system_prompt,
    build_sma_transform_hook_user_message,
    manifest_mapping_for_target,
    sma_transform_hook_item_id,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation.preview import (
    write_sma_transform_hook_preview_json,
)

__all__ = [
    "build_sma_transform_hook_system_prompt",
    "build_sma_transform_hook_user_message",
    "generate_sma_transform_hook_preview_rows_for_entity",
    "generate_sma_transform_hook_spec",
    "load_hook_specs_from_sma_preview_path",
    "manifest_mapping_for_target",
    "sma_transform_hook_item_id",
    "write_sma_transform_hook_preview_json",
]
