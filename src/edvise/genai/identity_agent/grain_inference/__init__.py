"""
Step 2 — Grain contract: prompts, validated LLM output schema, optional row dedupe helpers.

Use with Step 1 output from ``edvise.genai.identity_agent.profiling`` (`KeyProfile`).
"""
from . import prompt_builder, schemas, runner