"""
Prompts for LLM hook generation after HITL selects ``reentry: generate_hook``.

Grain: custom dedup / row policy hooks targeting ``DedupPolicy.hook_spec``.
Term: ``year_extractor`` / ``season_extractor`` (or split-column helpers) for ``TermOrderConfig.hook_spec``.
"""

from __future__ import annotations

import json
from typing import Any

from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain, HITLItem


def build_hook_generation_system_prompt(domain: HITLDomain) -> str:
    """
    System prompt for :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.HookSpec` JSON.
    """
    if domain == HITLDomain.IDENTITY_GRAIN:
        return _system_prompt_grain()
    if domain == HITLDomain.IDENTITY_TERM:
        return _system_prompt_term()
    raise ValueError(f"Hook generation not supported for domain {domain!r}")


def build_hook_generation_user_message(
    item: HITLItem,
    config_snippet: dict[str, Any],
) -> str:
    """
    User payload: HITL item fields plus the relevant config slice (grain or term).

    ``config_snippet`` should be shaped like
    ``{"grain_contract": {...}}`` or ``{"term_config": {...}}`` (term_config may be null).
    """
    payload = {
        "item_id": item.item_id,
        "institution_id": item.institution_id,
        "table": item.table,
        "domain": item.domain.value,
        "hook_group_id": item.hook_group_id,
        "hitl_question": item.hitl_question,
        "hitl_context": item.hitl_context,
        "target": item.target.model_dump(mode="json"),
        "config_snippet": config_snippet,
    }
    return json.dumps(payload, indent=2)


def extract_config_snippet_for_hook_item(
    identity_config: dict[str, Any],
    item: HITLItem,
) -> dict[str, Any]:
    """
    Pull ``datasets[table].grain_contract`` or ``datasets[table].term_config`` from resolver-shaped JSON.
    """
    ds = identity_config.get("datasets")
    if not isinstance(ds, dict) or item.table not in ds:
        raise ValueError(
            f"No datasets[{item.table!r}] in config (have {list(ds or {}).keys()!r})"
        )
    row = ds[item.table]
    if not isinstance(row, dict):
        raise ValueError(f"datasets[{item.table!r}] must be an object")
    if item.domain == HITLDomain.IDENTITY_GRAIN:
        gc = row.get("grain_contract")
        if gc is None:
            raise ValueError(f"grain_contract missing for table {item.table!r}")
        return {"grain_contract": gc}
    if item.domain == HITLDomain.IDENTITY_TERM:
        return {"term_config": row.get("term_config")}
    raise ValueError(f"Unsupported HITL domain {item.domain!r}")


def _system_prompt_grain() -> str:
    return """
You are a code-generation assistant for IdentityAgent **grain** (deduplication) hooks.

Respond with **one JSON object only** — no markdown fences, no preamble. The object must validate as HookSpec:

{
  "file": "<relative path e.g. pipelines/<institution_id>/helpers/dedup_hooks.py>",
  "functions": [
    {
      "name": "<python function name>",
      "signature": "<def name(...): ...>",
      "description": "<what it does>",
      "example_input": "<optional>",
      "example_output": "<optional>",
      "draft": "<optional: single expression or short body as a string>"
    }
  ]
}

Rules:
- Functions implement the policy implied by hitl_question / hitl_context and the grain_contract snippet.
- Prefer small, testable functions; pandas may be imported at module level in the real file.
- ``file`` should live under pipelines/<institution_id>/helpers/ for the given institution_id in the user JSON.
- Output must be parseable JSON (double quotes, no trailing commas).
""".strip()


def _system_prompt_term() -> str:
    return """
You are a code-generation assistant for IdentityAgent **term** normalization hooks.

Respond with **one JSON object only** — no markdown fences, no preamble. The object must validate as HookSpec:

{
  "file": "<relative path e.g. pipelines/<institution_id>/helpers/term_hooks.py>",
  "functions": [
    {
      "name": "year_extractor",
      "signature": "def year_extractor(term: str) -> int",
      "description": "<what it does>",
      "example_input": "<raw token from data>",
      "example_output": "<expected int year>",
      "draft": "<single Python expression>"
    },
    {
      "name": "season_extractor",
      "signature": "def season_extractor(term: str) -> str",
      "description": "<what it does>",
      "example_input": "<raw token>",
      "example_output": "<raw token matching a 'raw' key in season_map>",
      "draft": "<single Python expression>"
    }
  ]
}

Rules:
- season_extractor output must match raw keys in season_map when term_config uses a single term column.
- If term_config is null, infer requirements from hitl_context only.
- ``file`` should live under pipelines/<institution_id>/helpers/ for the given institution_id in the user JSON.
- Output must be parseable JSON (double quotes, no trailing commas).
""".strip()


__all__ = [
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "extract_config_snippet_for_hook_item",
]
