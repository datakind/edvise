"""Pydantic models and helpers for IdentityAgent term column config (``term_config``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    Institution term encoding consumed by add_term_order and add_term_labels.

    Source columns are mutually exclusive:
    - term_col: single column encoding both year and season (e.g. "2018FA", "Fall 2019")
    - year_col + season_col: pre-separated year and season columns

    season_map is an ordered list of raw tokens with canonical season labels; list position is
    chronological rank (1-indexed) within a calendar year.

    When term_extraction is "custom", hook_spec must describe drafted year_extractor and
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
    term_extraction: Literal["standard", "custom"]
    hook_spec: dict[str, Any] | None = Field(
        default=None,
        description="Required when term_extraction == 'custom'; null for standard extraction.",
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
    def _hook_spec_when_custom(self) -> TermOrderConfig:
        # standard hook_spec rule
        if self.term_extraction == "custom" and self.hook_spec is None:
            raise ValueError("hook_spec is required when term_extraction is 'custom'")
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
        description="Executable term config for add_term_order and add_term_labels, or null if not applicable.",
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


__all__ = [
    "CANONICAL_SEASONS",
    "SeasonMapEntry",
    "TermContract",
    "TermOrderConfig",
]
