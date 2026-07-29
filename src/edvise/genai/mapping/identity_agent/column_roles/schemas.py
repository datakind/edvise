"""Pydantic models for per-column semantic role classification (ColumnRolesAgent)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .file_kinds import FileKind


class ColumnRole(str, Enum):
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


class ColumnRolesLLMResponse(BaseModel):
    """Structural + completeness validation of the raw ColumnRolesAgent JSON.

    Completeness (every input column assigned exactly once) is enforced when the
    caller supplies ``expected_columns`` via validation context. Because failures
    surface as :class:`pydantic.ValidationError`, they are retried with a
    correction hint by :func:`edvise.utils.llm_utils.call_with_retry` — an omitted
    column no longer hard-fails the onboard run on the first response.
    """

    file_kind: FileKind
    file_kind_confidence: float = Field(..., ge=0.0, le=1.0)
    file_kind_rationale: str = ""
    assignments: list[ColumnRoleAssignment]
    low_confidence_columns: list[str] = Field(default_factory=list)

    @field_validator("file_kind", mode="before")
    @classmethod
    def _normalize_file_kind(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("file_kind_rationale", mode="before")
    @classmethod
    def _coerce_rationale(cls, value: Any) -> str:
        return str(value or "")

    @field_validator("low_confidence_columns", mode="before")
    @classmethod
    def _coerce_low_confidence(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return value

    @model_validator(mode="after")
    def _check_expected_columns(self, info: ValidationInfo) -> "ColumnRolesLLMResponse":
        expected = (info.context or {}).get("expected_columns")
        if expected is None:
            return self
        assigned = {a.column for a in self.assignments}
        missing = [c for c in expected if c not in assigned]
        if missing:
            raise ValueError(f"Missing role assignments for columns: {missing}")
        extra = assigned - set(expected)
        if extra:
            raise ValueError(f"Unexpected columns in assignments: {sorted(extra)}")
        return self


class ColumnRolesResult(BaseModel):
    institution_id: str
    dataset: str
    file_kind: FileKind = Field(
        ...,
        description="Semantic table class from ColumnRolesAgent (student, course, etc.).",
    )
    file_kind_confidence: float = Field(..., ge=0.0, le=1.0)
    file_kind_rationale: str = Field(
        default="",
        description="Brief reason for file_kind (audit only)",
    )
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
