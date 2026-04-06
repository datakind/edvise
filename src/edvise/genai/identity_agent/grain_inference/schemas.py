"""Pydantic models for IdentityAgent grain contract output (LLM-validated JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

# Below this confidence score, `hitl_flag` must be true (ambiguous grain / policy required).
IDENTITY_CONFIDENCE_HITL_THRESHOLD: float = 0.5

# Valid `dedup_policy.strategy` values (JSON must use these exact strings).
DedupStrategy = Literal["true_duplicate", "temporal_collapse", "no_dedup", "policy_required"]

# Detected / declared raw term string shapes (IdentityAgent); executor may only fully implement a subset.
TermFormat = Literal["YYYYTT", "Season_YYYY", "YYYYMM", "YYYY_YY"]

# Approved term utilities for :class:`TermOrderConfig` (HITL / preprocessing — not output by IdentityAgent prompts).
TERM_UTILITY_REGISTRY: dict[str, str] = {
    "extract_term_season_from_term_code": "YYYYTT → FALL/SPRING/SUMMER (e.g. '2018FA' → 'FALL'). Accepts custom season_mapping param.",
    "normalize_term_code": "'Fall 2020' / 'SP' / short season codes → FALL/SPRING/SUMMER.",
    "extract_academic_year_from_term_code": "YYYYTT → YYYY-YY academic year (e.g. '2018FA' → '2018-19').",
    "format_academic_year_from_calendar_year": "Integer or string calendar year → YYYY-YY (e.g. 2018 → '2018-19').",
    "parse_term_description": "'Season YYYY' string → datetime (e.g. 'Fall 2020' → 2020-09-01).",
    "parse_term_code_to_datetime": "YYYYTT → datetime (e.g. '2018FA' → 2018-09-01).",
    "term_season_from_datetime": "datetime → FALL/SPRING/SUMMER based on month bands.",
    "extract_year": "Extract first 4-digit year from any string (e.g. '2018FA' → '2018').",
}


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
    Institution term encoding for ``add_term_order`` and related enrichment.

    ``canonical_mapping`` maps raw tokens (YYYYTT suffixes like ``FA``, or season words like
    ``Fall``) to canonical season labels (``FALL``, ``SPRING``) used to derive sort order.

    Registry utility names must match the keys in :data:`TERM_UTILITY_REGISTRY` in this module.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    term_column: str
    term_format: TermFormat | None = Field(
        default="YYYYTT",
        description=(
            "Format detected from sample values: YYYYTT (2018FA), Season_YYYY (Fall 2020), "
            "YYYYMM, YYYY_YY; null if unrecognized — set new_utility_needed."
        ),
    )
    term_parser: str | None = Field(
        default=None,
        description="Approved registry utility for parsing/normalizing terms, or null.",
    )
    term_sort_utility: str | None = Field(
        default=None,
        description="Registry utility for sort-key derivation, or null.",
    )
    term_academic_year_utility: str | None = Field(
        default=None,
        description="Registry utility for academic-year labeling, or null.",
    )
    term_parser_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the selected parser utility (e.g. custom season code maps).",
    )
    canonical_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Maps raw tokens to FALL/SPRING/SUMMER/WINTER.",
    )
    unmapped_values: list[str] = Field(
        default_factory=list,
        description="Raw term values that could not be mapped; flagged for HITL.",
    )
    new_utility_needed: bool = Field(
        default=False,
        validation_alias=AliasChoices("new_utility_needed", "NEW_UTILITY_NEEDED"),
        description="True if no registry utility fits even with term_parser_params — block enrichment until HITL.",
    )
    outputs: TermOrderOutputs = Field(default_factory=TermOrderOutputs)


class IdentityGrainContract(BaseModel):
    """
    Grain contract for one institution dataset, produced by IdentityAgent.

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
    term_config: TermOrderConfig | None = Field(
        default=None,
        description=(
            "Optional; not produced by IdentityAgent prompts — set via HITL or preprocessing. "
            "Term column, utilities, and mappings for add_term_order after dedup. "
            "See edvise.genai.identity_agent.execution.apply_grain_term_order."
        ),
    )

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
