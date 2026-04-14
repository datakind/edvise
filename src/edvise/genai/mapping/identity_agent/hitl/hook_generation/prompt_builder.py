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
        "reviewer_note": item.reviewer_note,
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

Respond with **one JSON object only** — no markdown fences, no preamble. The object must validate as HookSpec (omit ``file`` — the pipeline assigns the module path):

{
  "functions": [
    {
      "name": "<python function name — must match the def>",
      "signature": "<optional copy of the def line for display only>",
      "description": "<what it does>",
      "example_input": "<optional free-form string: human-readable description of representative input / DataFrame shape>",
      "example_output": "<optional free-form string: what the function returns or effect on grain>",
      "draft": "<one complete syntactically valid Python function: full def line + indented body>"
    }
  ]
}

Rules:
- **draft** must be a **complete function definition** in one string: from ``def name(...)`` through the body (indented). It must be parseable as part of a Python module. Put any needed imports **inside** the function body unless you emit a second function whose draft is only imports (avoid that — prefer imports inside ``def``).
- **name** must exactly match the function name in **draft**.
- **signature** is optional metadata for reviewers; materialization does not use it.
- **example_input** / **example_output** are **documentation only** for reviewers (e.g. describe column layout, sample keys, or transformation intent). They are **not** used for automated smoke tests. Use plain-language or structured prose.
- hitl_context contains the raw data samples the agent was looking at when it raised the flag — use it to understand the data shape.
- If reviewer_note is present, treat it as the authoritative instruction and override any draft logic in config_snippet.
- If reviewer_note is absent, use hitl_context and config_snippet together to infer the correct implementation.
- Functions implement the policy implied by hitl_question / hitl_context and the grain_contract snippet.
- Do **not** include a ``file`` field; hook module paths are assigned by the pipeline after review.
- Output must be parseable JSON (double quotes, no trailing commas).
""".strip()


def _system_prompt_term() -> str:
    return """
You are a code-generation assistant for IdentityAgent **term** normalization hooks.

Respond with **one JSON object only** — no markdown fences, no preamble. The object must validate as HookSpec (omit ``file`` — the pipeline assigns the module path):

{
  "functions": [
    {
      "name": "year_extractor",
      "signature": "<optional; e.g. def year_extractor(term: str) -> int>",
      "description": "<what it does>",
      "example_input": "<JSON string: text that ast.literal_eval parses to your test input (int, str, tuple, etc.)>",
      "example_output": "<JSON string: text that ast.literal_eval parses to the matching return value>",
      "draft": "<complete def year_extractor(term: str) -> int:\\n    ...body...>"
    },
    {
      "name": "season_extractor",
      "signature": "<optional>",
      "description": "<what it does>",
      "example_input": "<JSON string literal repr>",
      "example_output": "<JSON string literal repr>",
      "draft": "<complete def season_extractor(term: str) -> str:\\n    ...body...>"
    }
  ]
}

Rules:
- **draft** for each function must be the **full** ``def`` block (signature + body), syntactically valid Python, not a bare expression.
- **name** must match the ``def`` name in **draft**.
- **signature** is optional display-only metadata.
- **Term only — machine smoke tests:** **example_input** and **example_output** must be JSON **strings** whose text is a Python literal accepted by ``ast.literal_eval`` (numbers, quoted strings, tuples, etc.). After materialize, the pipeline runs smoke tests: **wrong examples log warnings** (fix in config if needed); **exceptions from executing the draft** fail materialization. Omit either field only if you truly cannot provide a round-trippable literal (smoke test will skip that function).
- **Before responding:** Mentally run each **draft** on ``literal_eval(example_input)`` and confirm the return value equals ``literal_eval(example_output)``. They must agree; revise the draft or the example strings until they do.
- **Same representation as the body:** The smoke test coerces inputs with ``literal_eval`` then calls your function. If the body uses string slicing or indexing, ensure you are slicing the same conceptual value (watch ``int`` vs ``str`` and string positions). Do not set ``example_output`` to match prose in the **description** if that disagrees with what the **code** returns for the chosen ``example_input``.
- hitl_context contains the raw data samples the agent was looking at when it raised the flag — use it to understand the data shape.
- If reviewer_note is present, treat it as the authoritative instruction and override any draft logic in config_snippet.
- If reviewer_note is absent, use hitl_context and config_snippet together to infer the correct implementation.
- season_extractor output must match raw keys in season_map when term_config uses a single term column.
- If term_config is null, infer requirements from hitl_context only.
- Do **not** include a ``file`` field; hook module paths are assigned by the pipeline after review.
- Output must be parseable JSON (double quotes, no trailing commas).
""".strip()


__all__ = [
    "build_hook_generation_system_prompt",
    "build_hook_generation_user_message",
    "extract_config_snippet_for_hook_item",
]
