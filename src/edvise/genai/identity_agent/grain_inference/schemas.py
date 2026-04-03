"""Pydantic models for IdentityAgent grain contract output (LLM-validated JSON)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Below this confidence score, `hitl_flag` must be true (ambiguous grain / policy required).
IDENTITY_CONFIDENCE_HITL_THRESHOLD: float = 0.5


class DedupStrategy(str, Enum):
    true_duplicate = "true_duplicate"
    temporal_collapse = "temporal_collapse"
    no_dedup = "no_dedup"


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    strategy: DedupStrategy
    sort_by: str | None = None
    keep: Literal["first", "last"] | None = None
    notes: str = ""


class IdentityGrainContract(BaseModel):
    """
    Grain contract for one institution dataset, produced by IdentityAgent.

    ``post_clean_primary_key`` is the proposed ``unique_keys`` for this source
    table in the schema contract (after cleaning). ``join_keys_for_2a`` informs
    SchemaMappingAgent Step 2a join key reasoning.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, use_enum_values=True)

    institution_id: str
    table: str
    post_clean_primary_key: list[str] = Field(
        ...,
        description="Proposed unique key column names after cleaning (maps to schema contract unique_keys).",
    )
    dedup_policy: DedupPolicy
    cleaning_collapses_to_student_grain: bool
    row_selection_required: bool
    join_keys_for_2a: list[str]
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Agent confidence in the proposed grain and dedup policy (same 0.0–1.0 scale as "
            "Schema Mapping Agent). Drives HITL — scores below the documented threshold require "
            "hitl_flag true."
        ),
    )
    hitl_flag: bool
    hitl_question: str | None = None
    reasoning: str
    term_order_column: str | None = Field(
        default=None,
        description=(
            "Optional column name for add_term_order after dedup (normalized names). "
            "See edvise.genai.identity_agent.execution.apply_grain_term_order."
        ),
    )

    @property
    def unique_keys(self) -> list[str]:
        """Alias for ``post_clean_primary_key`` (schema contract naming)."""
        return self.post_clean_primary_key

    @model_validator(mode="after")
    def low_confidence_requires_hitl(self) -> IdentityGrainContract:
        if self.confidence < IDENTITY_CONFIDENCE_HITL_THRESHOLD and not self.hitl_flag:
            raise ValueError(
                f"hitl_flag must be true when confidence is below {IDENTITY_CONFIDENCE_HITL_THRESHOLD}"
            )
        return self
