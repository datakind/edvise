"""Pydantic models and helpers for IdentityAgent term column config (``term_config``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Detected / declared raw term string shapes (IdentityAgent); executor may only fully implement a subset.
TermFormat = Literal["YYYYTT", "Season_YYYY", "YYYYMM", "YYYY_YY"]

# Legacy registry utility names (optional HITL / preprocessing); not part of the current TermOrderConfig schema.
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


class TermOrderOutputs(BaseModel):
    """Which optional term columns to add after :func:`~edvise.genai.identity_agent.term_normalization.utilities.add_edvise_term_order`."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    term_sort_key: bool = Field(default=True, alias="_term_sort_key")
    term_canonical: bool = Field(default=True, alias="_term_canonical")
    term_academic_year: bool = Field(default=True, alias="_term_academic_year")


class TermOrderConfig(BaseModel):
    """
    Institution term encoding consumed by :func:`~edvise.genai.identity_agent.term_normalization.utilities.add_edvise_term_order`
    and :func:`~edvise.genai.identity_agent.term_normalization.utilities.add_edvise_term_labels` via the same
    ``term_config`` dict.

    ``season_map`` is an ordered list of raw tokens with canonical season labels; list position is
    chronological rank (1-indexed) within a calendar year. When ``term_extraction`` is
    ``custom``, ``hook_spec`` must describe drafted ``year_extractor`` and ``season_extractor``
    hooks for human review before execution.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    term_col: str = Field(description="Authoritative term column name.")
    season_map: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            'Ordered list of {"raw": str, "canonical": str}; position = chronological rank '
            "(1-indexed). canonical must be one of: FALL, SPRING, SUMMER, WINTER."
        ),
    )
    term_extraction: Literal["standard", "custom"]
    hook_spec: dict[str, Any] | None = Field(
        default=None,
        description="Required when term_extraction == 'custom'; null for standard extraction.",
    )

    @model_validator(mode="after")
    def _hook_spec_when_custom(self) -> TermOrderConfig:
        if self.term_extraction == "custom" and self.hook_spec is None:
            raise ValueError("hook_spec is required when term_extraction is 'custom'")
        return self
