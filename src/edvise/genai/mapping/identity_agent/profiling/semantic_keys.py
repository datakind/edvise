"""
Build semantic grain key column sets from ColumnRolesAgent output.

Purely algorithmic — no LLM. Templates mirror DOMAIN PRIORS in grain_inference/prompt.py.
"""

from __future__ import annotations

import logging

from edvise.genai.mapping.identity_agent.column_roles.schemas import (
    ColumnRole,
    ColumnRolesResult,
)

logger = logging.getLogger(__name__)


def _first(columns: list[str]) -> str | None:
    return columns[0] if columns else None


def build_semantic_key_column_sets(
    dataset: str,
    roles: ColumnRolesResult,
) -> list[list[str]]:
    """
    Return ordered semantic key column lists to profile before combinatorial search.

    Keys are deduplicated; order is stable (base grains first, then widenings).
    """
    learner = _first(roles.columns_with_role(ColumnRole.LEARNER_ID))
    term = _first(roles.columns_with_role(ColumnRole.TERM))
    course_id = _first(roles.columns_with_role(ColumnRole.COURSE_ID))
    program = _first(roles.columns_with_role(ColumnRole.PROGRAM))
    major = _first(roles.columns_with_role(ColumnRole.MAJOR))
    cohort = _first(roles.columns_with_role(ColumnRole.COHORT))

    keys: list[list[str]] = []
    kind = dataset.strip().lower()

    if kind == "student":
        if learner:
            keys.append([learner])
            for extra in (program, major, cohort):
                if extra:
                    keys.append([learner, extra])
            if program and major:
                keys.append([learner, program, major])
        else:
            logger.warning(
                "semantic_keys: student dataset %s missing learner_id — no base semantic keys",
                dataset,
            )

    elif kind == "course":
        if learner and course_id and term:
            keys.append([learner, course_id, term])
        if learner and course_id:
            keys.append([learner, course_id])
        if learner and term:
            keys.append([learner, term])
        if not keys and learner:
            keys.append([learner])
            logger.warning(
                "semantic_keys: course dataset %s missing course_id and/or term — partial keys only",
                dataset,
            )

    elif kind == "semester":
        if learner and term:
            keys.append([learner, term])
        elif term:
            keys.append([term])
        elif learner:
            keys.append([learner])

    else:
        if learner:
            keys.append([learner])
        logger.info(
            "semantic_keys: unknown dataset kind %r — using learner-only template if present",
            dataset,
        )

    return _dedupe_key_lists(keys)


def _dedupe_key_lists(keys: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for cols in keys:
        sig = tuple(cols)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(list(cols))
    return out
