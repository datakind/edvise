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

from edvise.genai.mapping.shared.hitl.hook_spec.schemas import (
    HookFunctionSpec,
    HookSpec,
)  # noqa: F401
from edvise.genai.mapping.identity_agent.utilities import concat_model_sources
from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD

CANONICAL_SEASONS = {"FALL", "SPRING", "SUMMER", "WINTER"}

# Calendar-year rank for season_map ordering validation (SPRING → SUMMER → FALL → WINTER).
CALENDAR_CHRONOLOGY_RANK: dict[str, int] = {
    "SPRING": 1,
    "SUMMER": 2,
    "FALL": 3,
    "WINTER": 4,
}


def season_map_chronology_error(
    season_map: list[SeasonMapEntry] | list[dict[str, str]],
) -> str | None:
    """
    Return an error message when ``season_map`` canonicals are not calendar-chronological.

    Duplicate canonicals are allowed (e.g. two SUMMER entries); only the canonical season's
    calendar position must be non-decreasing left-to-right. Empty maps are valid (hooks/HITL
    may populate later).
    """
    if not season_map:
        return None

    ranked: list[tuple[int, int, str]] = []
    for i, entry in enumerate(season_map):
        canonical = (
            entry.canonical if isinstance(entry, SeasonMapEntry) else entry["canonical"]
        ).upper()
        rank = CALENDAR_CHRONOLOGY_RANK[canonical]
        ranked.append((i + 1, rank, canonical))

    for j in range(1, len(ranked)):
        prev_pos, prev_rank, prev_canon = ranked[j - 1]
        pos, rank, canon = ranked[j]
        if rank < prev_rank:
            return (
                "season_map must be calendar-chronological "
                "(SPRING → SUMMER → FALL → WINTER): "
                f"{prev_canon} (position {prev_pos}) precedes {canon} (position {pos})"
            )
    return None


class SeasonMapEntry(BaseModel):
    """One entry in the ordered season_map list."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    raw: str = Field(
        description="Raw season token as it appears in the institution data (e.g. 'FA', '9', 'Fall')."
    )
    canonical: str = Field(
        description="Canonical season label. Must be one of: FALL, SPRING, SUMMER, WINTER."
    )

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
    exclude_tokens: list[str] = Field(
        default_factory=list,
        description=(
            "Optional raw value prefixes to drop before term ordering (case-insensitive). "
            "Rows whose term_col value, or year_col / season_col value on the split path, "
            "starts with one of these strings after stripping whitespace are removed. "
            "Populated by HITL TermResolution.exclude_tokens."
        ),
    )
    term_extraction: Literal["standard", "hook_required"]
    year_semantics: (
        Literal[
            "calendar_literal",
            "academic_year_prefix",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "How the extracted year relates to the calendar year of the term. This is about "
            "year MEANING, not season encoding: how the season is spelled (e.g. '2024SP', "
            "'Fall 2019', numeric period codes like '2025-20') is handled entirely by "
            "season_map / hooks and is irrelevant here. "
            "null / 'calendar_literal' (default): the extracted year IS the calendar year "
            "(e.g. '2024SP' = Spring 2024, 'Fall 2019' = Fall 2019). "
            "'academic_year_prefix': the extracted year is the academic-year start; FALL/WINTER "
            "keep the year, SPRING/SUMMER roll forward one calendar year "
            "(e.g. '2017SR' = Spring 2018, or a period code '2025-20' = Spring 2026). "
            "Applies to both combined term_col and split year_col/season_col configs."
        ),
    )
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
            raise ValueError(
                "hook_spec is required when term_extraction is 'hook_required'"
            )
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

    @model_validator(mode="after")
    def _season_map_calendar_chronological(self) -> TermOrderConfig:
        err = season_map_chronology_error(self.season_map)
        if err:
            raise ValueError(err)
        return self


class TermContract(BaseModel):
    """
    Term-stage output from IdentityAgent — term normalization config plus LLM provenance metadata.

    ``term_config`` is null when the table does not require term normalization
    (e.g. row_selection_required is false, or no term column found).

    This is **term** output only; grain produces
    :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.GrainContract` without term fields.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

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
        description=(
            "True when hook_spec requires human review, term column selection is uncertain, "
            "or unique values contain unrecognized tokens. "
            "When true, HITLItems are emitted in the top-level hitl_items list."
        )
    )
    reasoning: str = Field(
        description="2-3 sentence summary of term column selection and format inference."
    )

    @model_validator(mode="after")
    def low_confidence_requires_hitl(self) -> TermContract:
        if self.confidence <= PIPELINE_HITL_CONFIDENCE_THRESHOLD and not self.hitl_flag:
            raise ValueError(
                f"hitl_flag must be true when confidence is at or below "
                f"{PIPELINE_HITL_CONFIDENCE_THRESHOLD}"
            )
        return self


class InstitutionTermContract(BaseModel):
    """
    Single JSON artifact for one institution: all dataset-level :class:`TermContract` values
    from term **batch** mode (one LLM response covering every table).

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


def get_term_contract_schema_context(
    *, include_institution_envelope: bool = False
) -> str:
    """Python source for term-stage contract models; optionally the batch envelope."""
    models: list[type] = [SeasonMapEntry, TermOrderConfig, TermContract]
    if include_institution_envelope:
        models.append(InstitutionTermContract)
    return concat_model_sources(models)


__all__ = [
    "CALENDAR_CHRONOLOGY_RANK",
    "CANONICAL_SEASONS",
    "HookFunctionSpec",
    "HookSpec",
    "InstitutionTermContract",
    "SeasonMapEntry",
    "TermContract",
    "TermOrderConfig",
    "get_term_contract_schema_context",
    "season_map_chronology_error",
]
