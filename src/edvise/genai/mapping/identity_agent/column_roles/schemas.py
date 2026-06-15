"""Pydantic models for per-column semantic role classification (ColumnRolesAgent)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ColumnRole(StrEnum):
    """Semantic role of a raw column for grain profiling."""

    LEARNER_ID = "learner_id"
    TERM = "term"
    COURSE_ID = "course_id"
    PROGRAM = "program"
    MAJOR = "major"
    COHORT = "cohort"
    MEASURE = "measure"
    INDEX = "index"
    METADATA = "metadata"
    OTHER = "other"


class ColumnRoleAssignment(BaseModel):
    column: str
    role: ColumnRole
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(
        default="",
        description="Brief reason for the label (audit only; not used downstream)",
    )


class ColumnRolesResult(BaseModel):
    institution_id: str
    dataset: str
    assignments: list[ColumnRoleAssignment]
    low_confidence_columns: list[str] = Field(
        default_factory=list,
        description="Columns where confidence is below the agent threshold",
    )
    profiler_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues surfaced to profiling artifacts and logs",
    )
    fallback_applied: list[str] = Field(
        default_factory=list,
        description="Columns whose role was adjusted by deterministic fallback rules",
    )

    def role_for(self, column: str) -> ColumnRole | None:
        for a in self.assignments:
            if a.column == column:
                return a.role
        return None

    def columns_with_role(self, role: ColumnRole) -> list[str]:
        return [a.column for a in self.assignments if a.role == role]

    def learner_id_column(self) -> str | None:
        cols = self.columns_with_role(ColumnRole.LEARNER_ID)
        return cols[0] if cols else None

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
