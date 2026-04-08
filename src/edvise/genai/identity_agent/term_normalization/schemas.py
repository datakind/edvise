"""Pydantic models and helpers for IdentityAgent term column config (``term_config``)."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from edvise.genai.identity_agent.grain_inference.schemas import HookFunctionSpec, HookSpec  # noqa: F401

CANONICAL_SEASONS = {"FALL", "SPRING", "SUMMER", "WINTER"}


class SeasonMapEntry(BaseModel):
    """One entry in the ordered season_map list."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    raw: str = Field(description="Raw season token as it appears in the institution data (e.g. 'FA', '9', 'Fall').")
    canonical: str = Field(description="Canonical season label. Must be one of: FALL, SPRING, SUMMER, WINTER.")

    @model_validator(mode="after")
    def _validate_canonical(self) -> SeasonMapEntry:
        if self.canonical not in CANONICAL_SEASONS:
            raise ValueError(
                f"canonical must be one of {sorted(CANONICAL_SEASONS)}, got {self.canonical!r}"
            )
        return self


class TermOrderConfig(BaseModel):
    """
    Institution term encoding consumed by ``add_edvise_term_order`` (order + labels)
    and optionally ``add_edvise_term_labels`` when ``_year`` / ``_season`` already exist.

    Source columns are mutually exclusive:
    - term_col: single column encoding both year and season (e.g. "2018FA", "Fall 2019")
    - year_col + season_col: pre-separated year and season columns

    season_map is an ordered list of raw tokens with canonical season labels; list position is
    chronological rank (1-indexed) within a calendar year.

    When term_extraction is "hook_required", hook_spec must describe drafted year_extractor and
    season_extractor functions for human review before execution. hook_spec is never needed
    when year_col and season_col are provided.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    term_col: str | None = Field(
        default=None,
        description=(
            "Single column encoding both year and season (e.g. '2018FA', 'Fall 2019'). "
            "Mutually exclusive with year_col and season_col."
        ),
    )
    year_col: str | None = Field(
        default=None,
        description=(
            "Year column when year and season are provided as separate columns. "
            "Must be set together with season_col. Mutually exclusive with term_col."
        ),
    )
    season_col: str | None = Field(
        default=None,
        description=(
            "Season column when year and season are provided as separate columns. "
            "Must be set together with year_col. Mutually exclusive with term_col."
        ),
    )
    season_map: list[SeasonMapEntry] = Field(
        default_factory=list,
        description=(
            "Ordered list of season entries; position = chronological rank (1-indexed) "
            "within a calendar year. canonical must be one of: FALL, SPRING, SUMMER, WINTER."
        ),
    )
    term_extraction: Literal["standard", "hook_required"]
    hook_spec: HookSpec | None = Field(
        default=None,
        description=(
            "Required when term_extraction == 'hook_required'; null for standard extraction. "
            "Same HookSpec shape as DedupPolicy.hook_spec."
        ),
    )

    @model_validator(mode="after")
    def _source_columns_valid(self) -> TermOrderConfig:
        has_single = self.term_col is not None
        has_split = self.year_col is not None and self.season_col is not None
        has_partial_split = (self.year_col is None) != (self.season_col is None)

        if not has_single and not has_split:
            raise ValueError(
                "Either term_col or both year_col and season_col must be provided."
            )
        if has_single and (self.year_col is not None or self.season_col is not None):
            raise ValueError(
                "term_col is mutually exclusive with year_col and season_col. "
                "Provide either term_col or both year_col and season_col."
            )
        if has_partial_split:
            raise ValueError(
                "year_col and season_col must be provided together, not individually."
            )
        return self

    @model_validator(mode="after")
    def _hook_spec_when_hook_required(self) -> TermOrderConfig:
        # standard hook_spec rule
        if self.term_extraction == "hook_required" and self.hook_spec is None:
            raise ValueError("hook_spec is required when term_extraction is 'hook_required'")
        # split column constraints
        if self.year_col is not None and self.season_col is not None:
            if self.term_extraction != "standard":
                raise ValueError(
                    "term_extraction must be 'standard' when year_col and season_col are provided."
                )
            if self.hook_spec is not None:
                raise ValueError(
                    "hook_spec must be null when year_col and season_col are provided."
                )
        return self


class TermContract(BaseModel):
    """
    Pass 2 output from IdentityAgent — term normalization config plus LLM provenance metadata.

    ``term_config`` is null when the table does not require term normalization
    (e.g. row_selection_required is false, or no term column found).

    This is **pass 2** output only; pass 1 produces
    :class:`~edvise.genai.identity_agent.grain_inference.schemas.GrainContract` without term fields.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    table: str
    term_config: TermOrderConfig | None = Field(
        default=None,
        description="Executable term config for add_edvise_term_order (term order + labels), or null if not applicable.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in term column selection and format inference (0.0–1.0).",
    )
    hitl_flag: bool = Field(
        description="True when hook_spec requires human review, or term column selection is uncertain."
    )
    hitl_question: str | None = Field(
        default=None,
        description="Specific question for human reviewer. Required when hitl_flag is true.",
    )
    reasoning: str = Field(
        description="2-3 sentence summary of term column selection and format inference."
    )

    @model_validator(mode="after")
    def _hitl_question_when_flagged(self) -> TermContract:
        if self.hitl_flag and not self.hitl_question:
            raise ValueError("hitl_question is required when hitl_flag is true")
        return self

    @model_validator(mode="after")
    def low_confidence_requires_hitl(self) -> TermContract:
        from edvise.genai.identity_agent.grain_inference.schemas import (
            IDENTITY_CONFIDENCE_HITL_THRESHOLD,
        )

        if self.confidence < IDENTITY_CONFIDENCE_HITL_THRESHOLD and not self.hitl_flag:
            raise ValueError(
                f"hitl_flag must be true when confidence is below {IDENTITY_CONFIDENCE_HITL_THRESHOLD}"
            )
        return self


class InstitutionTermContract(BaseModel):
    """
    Single JSON artifact for one institution: all dataset-level :class:`TermContract` values
    from Pass 2 **batch** mode (one LLM response covering every table).

    Keys in ``datasets`` match logical dataset names (same keys as grain ``contracts_by_dataset``).
    Each :class:`TermContract` must have ``table`` equal to its map key and ``institution_id``
    equal to the envelope.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    institution_id: str
    datasets: dict[str, TermContract]

    @field_validator("datasets")
    @classmethod
    def non_empty_dataset_keys(
        cls, v: dict[str, TermContract]
    ) -> dict[str, TermContract]:
        for name in v:
            if not name.strip():
                raise ValueError("dataset name keys must be non-empty")
        return v

    @model_validator(mode="after")
    def contracts_match_institution_and_keys(self) -> InstitutionTermContract:
        for dname, c in self.datasets.items():
            if c.institution_id != self.institution_id:
                raise ValueError(
                    f"Dataset {dname!r}: contract institution_id {c.institution_id!r} "
                    f"does not match envelope institution_id {self.institution_id!r}"
                )
            if c.table != dname:
                raise ValueError(
                    f"Dataset map key {dname!r} must match TermContract.table {c.table!r}"
                )
        return self

    def contracts_by_dataset(self) -> dict[str, TermContract]:
        """Same mapping expected by §8 / ``term_contract_by_dataset`` style call sites."""
        return dict(self.datasets)


__all__ = [
    "CANONICAL_SEASONS",
    "HookFunctionSpec",
    "HookSpec",
    "InstitutionTermContract",
    "SeasonMapEntry",
    "TermContract",
    "TermOrderConfig",
]
