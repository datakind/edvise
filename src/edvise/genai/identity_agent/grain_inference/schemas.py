"""Pydantic models for IdentityAgent grain contract output (LLM-validated JSON)."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Below this confidence score, `hitl_flag` must be true (ambiguous grain / policy required).
IDENTITY_CONFIDENCE_HITL_THRESHOLD: float = 0.5

# Valid `dedup_policy.strategy` values (JSON must use these exact strings).
DedupStrategy = Literal[
    "true_duplicate", "temporal_collapse", "no_dedup", "policy_required"
]


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: DedupStrategy
    sort_by: str | None = None
    keep: Literal["first", "last"] | None = None
    notes: str = ""


class GrainContract(BaseModel):
    """
    Grain contract for one institution dataset from IdentityAgent **pass 1** (grain only).

    Term column config is produced in **pass 2** as :class:`~edvise.genai.identity_agent.term_normalization.schemas.TermContract`.

    When ``student_id_alias`` is set, ``post_clean_primary_key`` and ``join_keys_for_2a``
    should name that column **as in the pre-canonical-rename frame** (the same string as
    ``student_id_alias``), so dedup/join keys stay consistent; downstream cleaning renames
    it to ``student_id`` for the frozen schema contract.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    table: str
    student_id_alias: str | None = Field(
        default=None,
        description=(
            "Institution student-identifier column **as shown in the column list** (header-normalized, "
            "typically snake_case), e.g. student_id_randomized_datakind. Use null when the column "
            "is already student_id after normalization, or when this table's grain has no student "
            "identifier. Downstream cleaning maps this to canonical student_id once."
        ),
    )
    post_clean_primary_key: list[str] = Field(
        ...,
        description=(
            "Grain primary key column names aligned with the frame used for grain dedup: when "
            "student_id_alias is set, list that column name (not literal student_id) wherever the "
            "student identifier is part of the key. Maps to schema contract unique_keys after the "
            "canonical student_id rename."
        ),
    )
    dedup_policy: DedupPolicy
    row_selection_required: bool
    join_keys_for_2a: list[str] = Field(
        ...,
        description=(
            "Join keys for SchemaMappingAgent 2a; same naming convention as post_clean_primary_key "
            "for the student identifier column when student_id_alias is set."
        ),
    )
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
    notes: str = ""

    @property
    def unique_keys(self) -> list[str]:
        """Alias for ``post_clean_primary_key`` (schema contract naming)."""
        return self.post_clean_primary_key

    @field_validator("student_id_alias", mode="before")
    @classmethod
    def _empty_student_id_alias_to_none(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @model_validator(mode="after")
    def low_confidence_requires_hitl(self) -> GrainContract:
        if self.confidence < IDENTITY_CONFIDENCE_HITL_THRESHOLD and not self.hitl_flag:
            raise ValueError(
                f"hitl_flag must be true when confidence is below {IDENTITY_CONFIDENCE_HITL_THRESHOLD}"
            )
        return self


class InstitutionGrainContract(BaseModel):
    """
    Single JSON artifact for one institution: all dataset-level :class:`GrainContract` values.

    Keys in ``datasets`` match logical dataset names (same keys as ``inputs.toml`` / ``SchoolMappingConfig.datasets``).
    Use for testing or handoff instead of N separate per-table JSON files.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    datasets: dict[str, GrainContract]

    @field_validator("datasets")
    @classmethod
    def non_empty_dataset_keys(
        cls, v: dict[str, GrainContract]
    ) -> dict[str, GrainContract]:
        for name in v:
            if not name.strip():
                raise ValueError("dataset name keys must be non-empty")
        return v

    @model_validator(mode="after")
    def contracts_match_institution(self) -> InstitutionGrainContract:
        for dname, c in self.datasets.items():
            if c.institution_id != self.institution_id:
                raise ValueError(
                    f"Dataset {dname!r}: contract institution_id {c.institution_id!r} "
                    f"does not match envelope institution_id {self.institution_id!r}"
                )
        return self

    def contracts_by_dataset(self) -> dict[str, GrainContract]:
        """Return the same mapping as :func:`run_identity_agents_for_institution` / schema merge APIs expect."""
        return dict(self.datasets)


def build_institution_grain_contracts(
    institution_id: str,
    contracts_by_dataset: dict[str, GrainContract],
) -> InstitutionGrainContract:
    """Wrap per-dataset grain contracts in one envelope (single JSON file for testing or handoff)."""
    return InstitutionGrainContract(
        institution_id=institution_id, datasets=dict(contracts_by_dataset)
    )
