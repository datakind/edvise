"""Pydantic models for IdentityAgent table profiling (raw snapshot + candidate keys)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .constants import (
    SAMPLE_VALUES_TOP_N,
    UNIQUE_VALUES_MAX_CARDINALITY,
)


class RawColumnProfile(BaseModel):
    name: str
    dtype: str
    null_rate: float = Field(..., description="Fraction of null values")
    unique_count: int
    unique_values: list | None = Field(
        None,
        description=(
            f"All distinct values when unique_count <= {UNIQUE_VALUES_MAX_CARDINALITY}, else null"
        ),
    )
    sample_values: list = Field(
        ...,
        description=f"Top {SAMPLE_VALUES_TOP_N} most frequent non-null values",
    )
    is_term_candidate: bool = Field(
        False,
        description="True if column name matches known term column patterns",
    )


class RawTableProfile(BaseModel):
    institution_id: str
    dataset: str
    row_count: int
    column_count: int
    columns: list[RawColumnProfile]

    @property
    def term_candidates(self) -> list[RawColumnProfile]:
        """Columns flagged as term candidates for IdentityAgent term config inference."""
        return [c for c in self.columns if c.is_term_candidate]


class CandidateKey(BaseModel):
    columns: list[str]
    uniqueness_score: float = Field(
        ..., description="Fraction of rows unique on this key"
    )
    null_rate: float = Field(..., description="Max null rate across key columns")
    rank: int


class ColumnVarianceProfile(BaseModel):
    column: str
    pct_groups_with_variance: float = Field(
        ...,
        description="Fraction of non-unique groups where this column has >1 distinct value",
    )
    sample_values: list = Field(
        ...,
        description="Up to 5 sample values from non-unique rows for LLM context",
    )


class CandidateProfile(BaseModel):
    candidate_key: CandidateKey
    non_unique_rows: int
    affected_groups: int
    group_size_distribution: dict[int, int]
    within_group_variance: list[ColumnVarianceProfile] = Field(
        ...,
        description=(
            "Per non-key column: how often it varies across non-unique groups. "
            "Empty when key is fully unique. Interpretation (grain vs noise vs policy) "
            "is left to IdentityAgent (Step 2)."
        ),
    )
    sampled: bool = Field(
        False, description="True if variance profile is based on sampled groups"
    )


class RankedCandidateProfiles(BaseModel):
    candidate_key_profiles: list[CandidateProfile] = Field(
        ...,
        description=(
            "Profile per candidate key, ranked by uniqueness. Pass to "
            "``edvise.genai.mapping.identity_agent.grain_inference.prompt_builder."
            "build_identity_agent_user_message`` (Step 2)."
        ),
    )


class KeyProfileResult(BaseModel):
    """Bundled output of raw column profiling plus candidate key profiling."""

    raw_table_profile: RawTableProfile
    key_profile: RankedCandidateProfiles
