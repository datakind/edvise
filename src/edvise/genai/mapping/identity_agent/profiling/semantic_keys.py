"""
Build semantic grain key column sets from ColumnRolesAgent output.

Purely algorithmic — no LLM. Templates mirror DOMAIN PRIORS in grain_inference/prompt.py.
"""

from __future__ import annotations

import logging

from edvise.genai.mapping.identity_agent.column_roles.file_kinds import FileKind
from edvise.genai.mapping.identity_agent.column_roles.schemas import (
    ColumnRole,
    ColumnRolesResult,
)

logger = logging.getLogger(__name__)


def _first(columns: list[str]) -> str | None:
    return columns[0] if columns else None


def _course_id_columns(roles: ColumnRolesResult) -> list[str]:
    """All ``course_id`` columns in assignment order (composite identifiers)."""
    return roles.columns_with_role(ColumnRole.COURSE_ID)


def _append_student_keys(
    keys: list[list[str]],
    *,
    learner: str,
    program: str | None,
    major: str | None,
    cohort: str | None,
) -> None:
    keys.append([learner])
    for extra in (program, major, cohort):
        if extra:
            keys.append([learner, extra])
    if program and major:
        keys.append([learner, program, major])


def _append_degree_keys(
    keys: list[list[str]],
    *,
    learner: str | None,
    term: str | None,
    program: str | None,
    major: str | None,
) -> None:
    if learner and term:
        keys.append([learner, term])
    if learner and program and term:
        keys.append([learner, program, term])
    if learner and program:
        keys.append([learner, program])
    if learner and program and major:
        keys.append([learner, program, major])
    if learner and major and term:
        keys.append([learner, major, term])
    if learner:
        keys.append([learner])
    elif term:
        keys.append([term])


def _append_role_driven_fallback_keys(
    keys: list[list[str]],
    *,
    learner: str | None,
    term: str | None,
) -> None:
    """``other`` file kind: seed from roles when no template matched."""
    if learner and term:
        keys.append([learner, term])
    elif term:
        keys.append([term])
    elif learner:
        keys.append([learner])


def build_semantic_key_column_sets(
    dataset: str,
    roles: ColumnRolesResult,
) -> list[list[str]]:
    """
    Return ordered semantic key column lists to profile before combinatorial search.

    Uses ``roles.file_kind`` from ColumnRolesAgent. Keys are deduplicated; order is
    stable (base grains first, then widenings).
    """
    learner = _first(roles.columns_with_role(ColumnRole.LEARNER_ID))
    term = _first(roles.columns_with_role(ColumnRole.TERM))
    course_ids = _course_id_columns(roles)
    program = _first(roles.columns_with_role(ColumnRole.PROGRAM))
    major = _first(roles.columns_with_role(ColumnRole.MAJOR))
    cohort = _first(roles.columns_with_role(ColumnRole.COHORT))

    keys: list[list[str]] = []
    kind = roles.file_kind

    if kind == FileKind.STUDENT:
        if learner:
            _append_student_keys(
                keys,
                learner=learner,
                program=program,
                major=major,
                cohort=cohort,
            )
        else:
            logger.warning(
                "semantic_keys: student file %s missing learner_id — no base semantic keys",
                dataset,
            )

    elif kind == FileKind.COURSE:
        if learner and course_ids and term:
            keys.append([learner, *course_ids, term])
        if learner and course_ids:
            keys.append([learner, *course_ids])
        if learner and term:
            keys.append([learner, term])
        if not keys and learner:
            keys.append([learner])
            logger.warning(
                "semantic_keys: course file %s missing course_id and/or term — partial keys only",
                dataset,
            )

    elif kind == FileKind.SEMESTER:
        if learner and term:
            keys.append([learner, term])
        elif term:
            keys.append([term])
        elif learner:
            keys.append([learner])

    elif kind == FileKind.DEGREE:
        _append_degree_keys(
            keys,
            learner=learner,
            term=term,
            program=program,
            major=major,
        )

    else:
        _append_role_driven_fallback_keys(keys, learner=learner, term=term)
        if keys:
            logger.info(
                "semantic_keys: file_kind=other for %s — role-driven seeds %s",
                dataset,
                keys,
            )
        else:
            logger.info(
                "semantic_keys: file_kind=other for %s — no semantic keys (statistical search only)",
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
