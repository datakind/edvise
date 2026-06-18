"""Resolve semantic entity kind for profiling (decoupled from inputs.toml dataset keys)."""

from __future__ import annotations

CANONICAL_ENTITY_KINDS: frozenset[str] = frozenset(
    {"student", "course", "semester", "degree"}
)

# Backwards-compatible aliases when ``entity_kind`` is not set explicitly in inputs.toml.
ENTITY_KIND_ALIASES: dict[str, str] = {
    "cohort": "student",
    "students": "student",
    "registration": "course",
    "registrations": "course",
    "enrollment": "course",
    "enrollments": "course",
    "class": "course",
    "classes": "course",
    "section": "course",
    "sections": "course",
    "terms": "semester",
    "term_summary": "semester",
    "student_term": "semester",
    "student_terms": "semester",
    "award": "degree",
    "awards": "degree",
    "degrees": "degree",
    "conferral": "degree",
    "conferrals": "degree",
}


def resolve_entity_kind(
    dataset_name: str,
    *,
    configured_kind: str | None = None,
) -> str:
    """
    Map a logical dataset name to a semantic template kind.

    Priority:
    1. Explicit ``configured_kind`` from ``DatasetConfig.entity_kind`` / inputs.toml
    2. Exact canonical kind match on the dataset name (``student``, ``course``, …)
    3. Built-in alias table (``registration`` → ``course``, etc.)
    4. Lowercased dataset name unchanged (unknown template)
    """
    if configured_kind is not None and str(configured_kind).strip():
        key = str(configured_kind).strip().lower()
        if key in CANONICAL_ENTITY_KINDS:
            return key
        return ENTITY_KIND_ALIASES.get(key, key)

    key = dataset_name.strip().lower()
    if key in CANONICAL_ENTITY_KINDS:
        return key
    return ENTITY_KIND_ALIASES.get(key, key)


__all__ = [
    "CANONICAL_ENTITY_KINDS",
    "ENTITY_KIND_ALIASES",
    "resolve_entity_kind",
]
