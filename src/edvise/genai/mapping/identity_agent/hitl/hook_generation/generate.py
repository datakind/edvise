"""Orchestrate one LLM call per :class:`HITLItem` to produce a :class:`HookSpec`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLItem

from .parse import parse_hook_spec
from .paths import ensure_hook_spec_file
from .prompt_builder import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
)


def generate_hook_spec(
    *,
    item: HITLItem,
    identity_config: dict[str, Any],
    llm_complete: Callable[[str, str], str],
    normalized_columns: list[str] | None = None,
) -> HookSpec:
    """
    Build hook-generation prompts for ``item.domain``, call ``llm_complete(system, user)``, parse ``HookSpec``.

    ``identity_config`` must match resolver-shaped JSON (``datasets[table].grain_contract`` / ``term_config``).

    ``normalized_columns``: optional list of column names on the cleaned dataframe (after
    ``normalize_columns``). Pass ``normalized_column_names_from_raw_headers`` from
    ``hook_generation.prompt_builder`` on raw file headers, or omit if unavailable.
    """
    snippet = extract_config_snippet_for_hook_item(identity_config, item)
    system = build_hook_generation_system_prompt(item.domain)
    user = build_hook_generation_user_message(
        item, snippet, normalized_columns=normalized_columns
    )
    raw = llm_complete(system, user)
    spec = parse_hook_spec(raw)
    return ensure_hook_spec_file(
        spec, institution_id=item.institution_id, domain=item.domain
    )


def generate_hook_specs_for_hook_items(
    *,
    hitl_path: str | Path,
    config_path: str | Path,
    llm_complete: Callable[[str, str], str],
    normalized_columns_by_table: dict[str, list[str]] | None = None,
) -> list[tuple[str, HookSpec]]:
    """
    Run :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.get_hook_items` and generate a spec per representative item.

    Returns (item_id, HookSpec) pairs. Call
    :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.apply_hook_spec` for each pair;
    pass ``materialize=True`` and ``repo_root=`` (typically ``bronze_volumes_path``) to emit the
    hook module at ``hook_spec.file``.

    ``normalized_columns_by_table``: optional map ``dataset_name`` → column names after
    ``normalize_columns`` (build with ``normalized_column_names_from_raw_headers``).
    Keys must match each hook item's ``table`` field.
    """
    # Local import avoids circular imports at package load time.
    from edvise.genai.mapping.identity_agent.hitl.resolver import get_hook_items

    hitl_path = Path(hitl_path)
    config_path = Path(config_path)
    items = get_hook_items(hitl_path)
    config = json.loads(config_path.read_text())
    out: list[tuple[str, HookSpec]] = []
    for item in items:
        nc: list[str] | None = None
        if normalized_columns_by_table:
            nc = normalized_columns_by_table.get(item.table)
        spec = generate_hook_spec(
            item=item,
            identity_config=config,
            llm_complete=llm_complete,
            normalized_columns=nc,
        )
        out.append((item.item_id, spec))
    return out


__all__ = [
    "generate_hook_spec",
    "generate_hook_specs_for_hook_items",
]
