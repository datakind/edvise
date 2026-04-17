"""
Step 2a manifest prompts.

- :mod:`edvise.genai.mapping.schema_mapping_agent.manifest.prompts.generate` —
  original 2a LLM prompts (single-pass, batched, two-pass).
- :mod:`edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine` —
  refinement + HITL prompts after deterministic validation.
"""

from __future__ import annotations

from .generate import (
    assemble_step2a_batched_prompt_from_sections,
    assemble_step2a_cohort_pass_prompt_from_sections,
    assemble_step2a_course_pass_prompt_from_sections,
    assemble_step2a_prompt_from_sections,
    audit_step2a_batched_prompt,
    audit_step2a_prompt,
    build_step2a_batched_prompt,
    build_step2a_prompt,
    build_step2a_prompt_cohort_pass,
    build_step2a_prompt_course_pass,
    collect_step2a_prompt_batched_sections,
    collect_step2a_prompt_cohort_pass_sections,
    collect_step2a_prompt_course_pass_sections,
    collect_step2a_prompt_sections,
    extract_schema_descriptor,
    load_json,
    merge_step2a_entity_manifests,
    slim_reference_manifest,
    strip_json_fences,
    summarize_schema_contract,
)
from .refine import (
    build_refinement_combined_system_prompt,
    build_refinement_combined_user_prompt,
    build_refinement_system_prompt,
    build_refinement_user_prompt,
)

__all__ = [
    "assemble_step2a_batched_prompt_from_sections",
    "assemble_step2a_cohort_pass_prompt_from_sections",
    "assemble_step2a_course_pass_prompt_from_sections",
    "assemble_step2a_prompt_from_sections",
    "audit_step2a_batched_prompt",
    "audit_step2a_prompt",
    "build_refinement_combined_system_prompt",
    "build_refinement_combined_user_prompt",
    "build_refinement_system_prompt",
    "build_refinement_user_prompt",
    "build_step2a_batched_prompt",
    "build_step2a_prompt",
    "build_step2a_prompt_cohort_pass",
    "build_step2a_prompt_course_pass",
    "collect_step2a_prompt_batched_sections",
    "collect_step2a_prompt_cohort_pass_sections",
    "collect_step2a_prompt_course_pass_sections",
    "collect_step2a_prompt_sections",
    "extract_schema_descriptor",
    "load_json",
    "merge_step2a_entity_manifests",
    "slim_reference_manifest",
    "strip_json_fences",
    "summarize_schema_contract",
]
