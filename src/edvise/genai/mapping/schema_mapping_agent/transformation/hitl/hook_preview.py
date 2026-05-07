"""SMA transform HookSpec preview JSON (same envelope shape as IA hook previews)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_sma_transform_hook_preview_json(
    *,
    output_path: str | Path,
    institution_id: str,
    domain: str,
    spec_rows: list[dict[str, Any]],
) -> None:
    """
    Write preview artifact for UC phase ``sma_gate_2_hook_preview``.

    ``domain`` should be ``schema_mapping_transform_cohort`` or
    ``schema_mapping_transform_course`` (informative for reviewers / Streamlit).

    Each row: ``item_id``, ``hook_spec`` (dict), optional ``review_context``.
    """
    path = Path(output_path)
    payload: dict[str, Any] = {
        "institution_id": institution_id,
        "domain": domain,
        "specs": spec_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


__all__ = ["write_sma_transform_hook_preview_json"]
