"""Pydantic models for IdentityAgent grain contract output (LLM-validated JSON)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Below this confidence score, `hitl_flag` must be true (ambiguous grain / policy required).
IDENTITY_CONFIDENCE_HITL_THRESHOLD: float = 0.5

# Valid `dedup_policy.strategy` values (JSON must use these exact strings).
DedupStrategy = Literal["true_duplicate", "temporal_collapse", "no_dedup"]


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: DedupStrategy
    sort_by: str | None = None
    keep: Literal["first", "last"] | None = None
    notes: str = ""


class TermOrderOutputs(BaseModel):
    """Which optional term columns to add after :func:`~edvise.feature_generation.term.add_term_order`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    term_sort_key: bool = Field(default=True, alias="_term_sort_key")
    term_canonical: bool = Field(default=True, alias="_term_canonical")
    term_academic_year: bool = Field(default=True, alias="_term_academic_year")


class TermOrderConfig(BaseModel):
    """
    Institution term encoding for ``add_term_order`` (e.g. ``YYYYTT`` codes + season aliases).

    ``canonical_mapping`` maps short suffix codes (``FA``, ``SP``) to canonical season labels
    (``FALL``, ``SPRING``) used to derive ``season_order`` via ``DEFAULT_SEASON_ORDER_MAP``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    term_column: str
    term_format: Literal["YYYYTT"] = "YYYYTT"
    canonical_mapping: dict[str, str] = Field(default_factory=dict)
    unmapped_values: list[str] = Field(default_factory=list)
    outputs: TermOrderOutputs = Field(default_factory=TermOrderOutputs)


class IdentityGrainContract(BaseModel):
    """
    Grain contract for one institution dataset, produced by IdentityAgent.

    ``post_clean_primary_key`` is the proposed ``unique_keys`` for this source
    table in the schema contract (after cleaning). ``join_keys_for_2a`` informs
    SchemaMappingAgent Step 2a join key reasoning.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    table: str
    post_clean_primary_key: list[str] = Field(
        ...,
        description="Proposed unique key column names after cleaning (maps to schema contract unique_keys).",
    )
    dedup_policy: DedupPolicy
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
    term_config: TermOrderConfig | None = Field(
        default=None,
        description=(
            "Optional term column + YYYYTT-style mapping for add_term_order after dedup. "
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


class InstitutionGrainContracts(BaseModel):
    """
    Single JSON artifact for one institution: all dataset-level :class:`IdentityGrainContract` values.

    Keys in ``datasets`` match logical dataset names (same keys as ``inputs.toml`` / ``SchoolMappingConfig.datasets``).
    Use for testing or handoff instead of N separate per-table JSON files.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    datasets: dict[str, IdentityGrainContract]

    @field_validator("datasets")
    @classmethod
    def non_empty_dataset_keys(cls, v: dict[str, IdentityGrainContract]) -> dict[str, IdentityGrainContract]:
        for name in v:
            if not name.strip():
                raise ValueError("dataset name keys must be non-empty")
        return v

    @model_validator(mode="after")
    def contracts_match_institution(self) -> InstitutionGrainContracts:
        for dname, c in self.datasets.items():
            if c.institution_id != self.institution_id:
                raise ValueError(
                    f"Dataset {dname!r}: contract institution_id {c.institution_id!r} "
                    f"does not match envelope institution_id {self.institution_id!r}"
                )
        return self

    def contracts_by_dataset(self) -> dict[str, IdentityGrainContract]:
        """Return the same mapping as :func:`run_identity_agents_for_institution` / schema merge APIs expect."""
        return dict(self.datasets)


def build_institution_grain_contracts(
    institution_id: str,
    contracts_by_dataset: dict[str, IdentityGrainContract],
) -> InstitutionGrainContracts:
    """Wrap per-dataset contracts in one envelope (single JSON file for testing or handoff)."""
    return InstitutionGrainContracts(institution_id=institution_id, datasets=dict(contracts_by_dataset))
