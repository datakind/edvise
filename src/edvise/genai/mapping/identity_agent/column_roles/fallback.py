"""Deterministic fallbacks when ColumnRolesAgent is uncertain on critical roles."""

from __future__ import annotations

import logging

from edvise.genai.mapping.identity_agent.profiling.constants import (
    DEVELOPMENTAL_MEASURE_COLUMN_PATTERNS,
    INDEX_COLUMN_PATTERNS,
    STUDENT_ANCHOR_NAME_PATTERN,
    TERM_COLUMN_PATTERNS,
)

from .prompt import COLUMN_ROLES_CONFIDENCE_THRESHOLD
from .schemas import ColumnRole, ColumnRoleAssignment, ColumnRolesResult

logger = logging.getLogger(__name__)

_OMITTED_COLUMN_CONFIDENCE = 0.5


def _guess_role_for_omitted_column(column: str) -> tuple[ColumnRole, str]:
    """Best-effort semantic role when the LLM omitted a column entirely."""
    name = str(column)
    if INDEX_COLUMN_PATTERNS.match(name):
        return ColumnRole.INDEX, "index column pattern"
    if STUDENT_ANCHOR_NAME_PATTERN.search(name):
        return ColumnRole.LEARNER_ID, "student-anchor name pattern"
    if TERM_COLUMN_PATTERNS.search(name):
        return ColumnRole.TERM, "term name pattern"
    if DEVELOPMENTAL_MEASURE_COLUMN_PATTERNS.search(name):
        return ColumnRole.MEASURE, "developmental/placement measure pattern"
    return ColumnRole.OTHER, "no name-pattern match for omitted column"


def _assignment_map(result: ColumnRolesResult) -> dict[str, ColumnRoleAssignment]:
    return {a.column: a for a in result.assignments}


def apply_column_role_fallbacks(
    result: ColumnRolesResult,
    *,
    columns: list[str],
) -> ColumnRolesResult:
    """
    Patch critical roles using name heuristics when the LLM omitted or mislabeled them.

    Mutates assignments in-place on a copied result; records warnings and ``fallback_applied``.
    """
    assignments = _assignment_map(result)
    warnings = list(result.profiler_warnings)
    fallback_applied: list[str] = []
    low_confidence = set(result.low_confidence_columns)

    def _set_role(
        column: str,
        role: ColumnRole,
        *,
        reason: str,
        confidence: float = 0.75,
    ) -> None:
        prev = assignments.get(column)
        if (
            prev
            and prev.role == role
            and prev.confidence >= COLUMN_ROLES_CONFIDENCE_THRESHOLD
        ):
            return
        assignments[column] = ColumnRoleAssignment(
            column=column,
            role=role,
            confidence=confidence,
            rationale=f"fallback: {reason}",
        )
        fallback_applied.append(column)
        low_confidence.discard(column)
        warnings.append(f"column_roles_fallback: {column} -> {role.value} ({reason})")
        logger.info("ColumnRoles fallback: %s -> %s (%s)", column, role.value, reason)

    def _columns_with_role(role: ColumnRole) -> list[str]:
        return [c for c, a in assignments.items() if a.role == role]

    learner_cols = _columns_with_role(ColumnRole.LEARNER_ID)
    learner_ok = bool(learner_cols) and all(
        assignments[c].confidence >= COLUMN_ROLES_CONFIDENCE_THRESHOLD
        for c in learner_cols
    )
    if not learner_ok:
        for col in columns:
            if STUDENT_ANCHOR_NAME_PATTERN.search(str(col)):
                _set_role(
                    col, ColumnRole.LEARNER_ID, reason="student-anchor name pattern"
                )
                break
        if not _columns_with_role(ColumnRole.LEARNER_ID):
            warnings.append(
                "column_roles: no learner_id column identified (LLM + fallback failed)"
            )

    term_cols = _columns_with_role(ColumnRole.TERM)
    term_ok = bool(term_cols) and all(
        assignments[c].confidence >= COLUMN_ROLES_CONFIDENCE_THRESHOLD
        for c in term_cols
    )
    if not term_ok:
        for col in columns:
            if TERM_COLUMN_PATTERNS.search(str(col)):
                _set_role(col, ColumnRole.TERM, reason="term name pattern")
                break

    for col in columns:
        if INDEX_COLUMN_PATTERNS.match(str(col)) and assignments.get(col, None) is None:
            _set_role(
                col, ColumnRole.INDEX, reason="index column pattern", confidence=0.95
            )

    for col in columns:
        if col in assignments:
            continue
        role, reason = _guess_role_for_omitted_column(col)
        assignments[col] = ColumnRoleAssignment(
            column=col,
            role=role,
            confidence=_OMITTED_COLUMN_CONFIDENCE,
            rationale=f"fallback: omitted from LLM response ({reason})",
        )
        fallback_applied.append(col)
        low_confidence.add(col)
        warnings.append(
            f"column_roles_fallback: {col} -> {role.value} "
            f"(omitted from LLM response; {reason})"
        )
        logger.info(
            "ColumnRoles fallback: %s -> %s (omitted from LLM response; %s)",
            col,
            role.value,
            reason,
        )

    ordered = [assignments[c] for c in columns]
    return ColumnRolesResult(
        institution_id=result.institution_id,
        dataset=result.dataset,
        file_kind=result.file_kind,
        file_kind_confidence=result.file_kind_confidence,
        file_kind_rationale=result.file_kind_rationale,
        assignments=ordered,
        low_confidence_columns=sorted(low_confidence),
        profiler_warnings=warnings,
        fallback_applied=sorted(set(result.fallback_applied + fallback_applied)),
    )
