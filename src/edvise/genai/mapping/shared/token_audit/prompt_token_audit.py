"""
Local, API-free token **estimates** for Edvise prompt builders.

Uses a rough heuristic: ``estimated_tokens ≈ len(text) // CHARS_PER_TOKEN`` (often cited
as ~4 characters per token for English). Useful for comparing **relative** section size
and spotting bloated blocks—**not** for billing.

Sections are rolled into coarse buckets: ``system``, ``schema``, ``examples``,
``utilities``, ``user_query`` (see :func:`bucket_for_section_key`).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4

BUCKET_NAMES: tuple[str, ...] = (
    "system",
    "schema",
    "examples",
    "utilities",
    "user_query",
)

BucketName = Literal["system", "schema", "examples", "utilities", "user_query"]


def estimate_tokens(text: str, *, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """Rough token count: character length divided by ``chars_per_token`` (default 4)."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return len(text) // chars_per_token


def bucket_for_section_key(section_key: str) -> BucketName:
    """
    Map a section label (from our prompt builders) to a coarse bucket for bloat comparison.

    Rules are applied in order; first match wins.
    """
    k = section_key.lower()

    if "transformation_utilities" in k or k == "utilities":
        return "utilities"

    if "reference_manifest" in k or "reference_transformation" in k:
        return "examples"

    # Injected contracts / schema references (including nested under system.*)
    if any(
        sub in k
        for sub in (
            "pydantic",
            "schema_contract",
            "manifest_schema_reference",
            "target_schema",
            "transformation_map_schema",
            "grain_contract_schema",
            "hitl_item_schema",
            "term_contract_schema",
        )
    ):
        return "schema"

    if k.startswith("system.") or k == "system":
        return "system"

    return "user_query"


def audit_prompt_sections(
    sections: dict[str, str],
    *,
    builder: str,
    institution_id: str | None = None,
    institution_name: str | None = None,
    dataset_name: str | None = None,
    log: bool = True,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> dict[str, Any]:
    """
    Estimate tokens per section, aggregate into buckets, log, and return a summary dict.

    ``total_estimated_tokens`` is the sum of per-section estimates (sections should
    partition the prompt for a meaningful total).
    """
    buckets: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    sections_out: dict[str, int] = {}
    ordered: list[dict[str, Any]] = []

    for name, text in sections.items():
        n = estimate_tokens(text, chars_per_token=chars_per_token)
        b = bucket_for_section_key(name)
        buckets[b] += n
        sections_out[name] = n
        ordered.append({"section": name, "bucket": b, "estimated_tokens": n})
        if log:
            logger.info(
                "[prompt_token_audit] builder=%s section=%s bucket=%s est_tokens=%s "
                "institution_id=%s institution_name=%s dataset=%s",
                builder,
                name,
                b,
                n,
                institution_id,
                institution_name,
                dataset_name,
            )

    total_estimated_tokens = sum(sections_out.values())

    if log:
        logger.info(
            "[prompt_token_audit] builder=%s buckets=%s total_estimated_tokens=%s "
            "institution_id=%s institution_name=%s dataset=%s",
            builder,
            {k: buckets[k] for k in BUCKET_NAMES},
            total_estimated_tokens,
            institution_id,
            institution_name,
            dataset_name,
        )

    return {
        "estimator": "chars_div_n",
        "chars_per_token": chars_per_token,
        "builder": builder,
        "institution_id": institution_id,
        "institution_name": institution_name,
        "dataset_name": dataset_name,
        "sections": sections_out,
        "sections_ordered": ordered,
        "buckets": {k: buckets[k] for k in BUCKET_NAMES},
        "total_estimated_tokens": total_estimated_tokens,
        # Back-compat alias
        "section_sum_tokens": total_estimated_tokens,
    }
