"""
Prompts for LLM hook generation after HITL selects ``reentry: generate_hook``.

Grain: custom dedup / row policy hooks targeting ``DedupPolicy.hook_spec``.
Term: ``year_extractor`` / ``season_extractor`` (or split-column helpers) for ``TermOrderConfig.hook_spec``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from edvise.data_audit.custom_cleaning import normalize_columns
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain, HITLItem


def normalized_column_names_from_raw_headers(cols: Iterable[str]) -> list[str]:
    """
    Column names as they appear **after** :func:`~edvise.data_audit.custom_cleaning.normalize_columns`
    (same as step 1 of :func:`~edvise.data_audit.custom_cleaning.clean_dataset`).

    Pass raw file / CSV headers; use the returned list as ``normalized_columns`` for
    :func:`build_hook_generation_user_message` / :func:`generate_hook_spec` so the LLM sees
    authoritative names (e.g. ``major`` not ``Major``).
    """
    norm, _ = normalize_columns(cols)
    return [str(x) for x in norm]


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
    *,
    normalized_columns: list[str] | None = None,
) -> str:
    """
    User payload: HITL item fields plus the relevant config slice (grain or term).

    ``config_snippet`` should be shaped like
    ``{"grain_contract": {...}}`` or ``{"term_config": {...}}`` (term_config may be null).

    ``normalized_columns``: exact column names on the dataframe **after** ``normalize_columns``
    in ``clean_dataset`` (snake_case). Pass :func:`normalized_column_names_from_raw_headers` on
    raw headers, or any list aligned with the cleaned frame. When set, prompts tell the model to
    use **only** these names for indexing (avoids ``KeyError`` from raw header spellings in hitl_context).
    """
    payload: dict[str, Any] = {
        "item_id": item.item_id,
        "institution_id": item.institution_id,
        "table": item.table,
        "domain": item.domain.value,
        "hook_group_id": item.hook_group_id,
        "hook_group_tables": item.hook_group_tables,
        "hitl_question": item.hitl_question,
        "hitl_context": item.hitl_context,
        "reviewer_note": item.reviewer_note,
        "target": item.target.model_dump(mode="json"),
        "config_snippet": config_snippet,
    }
    if normalized_columns is not None:
        payload["normalized_columns"] = list(normalized_columns)
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
      "description": "<what it does — intent for reviewers>",
      "draft": "<one complete syntactically valid Python function: full def line + indented body>"
    }
  ]
}

Rules:
- **draft** must be a **complete function definition** in one string: from ``def name(...)`` through the body (indented). It must be parseable as part of a Python module. Put any needed imports **inside** the function body unless you emit a second function whose draft is only imports (avoid that — prefer imports inside ``def``).
- **Annotations vs in-function imports:** The ``def`` line’s parameter and return annotations are evaluated when the function is **defined** (module import time), **before** the body runs. If ``import pandas as pd`` / ``numpy`` / etc. appears only **inside** the body, do **not** write unquoted ``pd.DataFrame``, ``np.ndarray``, … on the ``def`` line — that raises ``NameError`` when the hook module loads. Either use **quoted** annotations (e.g. ``def f(group: "pd.DataFrame") -> "pd.DataFrame":``) and keep imports inside the body, or use built-in types only in annotations (e.g. omit hints or use ``typing.Any``).
- **name** must exactly match the function name in **draft**.
- **signature** is optional metadata for reviewers; materialization does not use it.
- **description** documents intent for reviewers. Materialization validates **draft** with syntax check (``ast.parse``) and optional pyflakes.
- hitl_context contains the raw data samples the agent was looking at when it raised the flag — use it to understand the data shape.
- **Column names in ``draft`` (e.g. ``group["…"]``) must match the cleaned dataframe:** headers are **normalized to snake_case** (``normalize_columns``) **before** ``dedupe_fn`` runs inside ``clean_dataset``.
- **When the user message JSON includes ``normalized_columns``:** treat that list as **authoritative** — use **only** those strings for bracket indexing and column logic. Do not infer names from ``hitl_context`` sample labels if they disagree (raw ``Major`` vs cleaned ``major``). If ``normalized_columns`` is absent, infer snake_case from context and config_snippet; wrong names cause ``KeyError`` at runtime.
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
      "description": "<what it does — intent for reviewers>",
      "draft": "<complete def year_extractor(term: str) -> int:\\n    ...body...>"
    },
    {
      "name": "season_extractor",
      "signature": "<optional>",
      "description": "<what it does — intent for reviewers>",
      "draft": "<complete def season_extractor(term: str) -> str:\\n    ...body...>"
    }
  ]
}

Rules:
- **draft** for each function must be the **full** ``def`` block (signature + body), syntactically valid Python, not a bare expression.
- **Annotations vs in-function imports:** Same as grain hooks: if imports (e.g. ``pandas``) are only inside the function body, use **quoted** annotations for ``pd.`` / ``np.`` types on the ``def`` line, or built-in-only annotations — never unquoted ``pd.DataFrame`` on the signature with ``import pandas`` below it.
- **name** must match the ``def`` name in **draft**.
- **signature** is optional display-only metadata.
- **description** documents intent for reviewers. After materialize, validation is **syntax** (``ast.parse``), optional **pyflakes**, then **signature** comparison in ``validate_hook`` (draft vs imported function) — no execution smoke tests.
- hitl_context contains the raw data samples the agent was looking at when it raised the flag — use it to understand the data shape.
- **When the user message JSON includes ``normalized_columns``:** those are the exact column names on the cleaned frame (after ``normalize_columns``). Use them for any code that references dataframe columns (e.g. if you read auxiliary columns beyond the term string). If absent, align with term_col / year_col / season_col from config_snippet using snake_case names.
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
    "normalized_column_names_from_raw_headers",
]
