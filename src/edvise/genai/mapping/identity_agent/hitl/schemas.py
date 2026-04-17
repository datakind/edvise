"""
Pydantic models for IdentityAgent HITL items.

HITLItem is the unit of human review — one item per ambiguity, per table.
Reviewer sets ``choice`` to a 1-based index into ``options`` (2–5 options per item).
hitl_resolver.py reads these files and applies the selected resolution to the
relevant config (grain_contract or term_config).

Scope: IdentityAgent Pass 1 (grain) and Pass 2 (term) only for M2.
SCHEMA_MAPPING and TRANSFORM domains are stubs for future use.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookFunctionSpec,
    HookSpec,
)
from edvise.genai.mapping.identity_agent.utilities import concat_model_sources
from edvise.genai.mapping.shared.hitl.run_log import RunEvent, RunLog


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HITLDomain(str, Enum):
    IDENTITY_GRAIN = "identity_grain"
    IDENTITY_TERM = "identity_term"
    SCHEMA_MAPPING = "schema_mapping"  # future
    TRANSFORM = "transform"  # future


class ReentryDepth(str, Enum):
    TERMINAL = "terminal"  # apply resolution, continue to next stage
    GENERATE_HOOK = "generate_hook"  # trigger hook gen LLM call, then continue


# ---------------------------------------------------------------------------
# GrainResolution — Pass 1 output mutations
# ---------------------------------------------------------------------------


class GrainResolution(BaseModel):
    """
    Mutations applied to the grain contract when a Pass 1 HITL item is resolved.

    dedup_strategy excludes 'policy_required' — that is the current state that
    triggered the HITL item, never a valid resolution target.

    When hook_spec is present, :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.resolve_items`
    writes ``dedup_policy.hook_spec`` and sets ``strategy='policy_required'`` (same as
    :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.apply_hook_spec`).

    All fields optional — resolver applies only those present.
    """

    candidate_key_override: list[str] | None = Field(
        default=None,
        description="Reviewer-corrected columns forming the post-clean primary key.",
    )
    dedup_strategy: (
        Literal[
            "true_duplicate",
            "temporal_collapse",
            "no_dedup",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Resolved dedup strategy. 'policy_required' is intentionally excluded — "
            "it is a flag state, not a resolution."
        ),
    )
    dedup_sort_by: str | None = Field(
        default=None,
        description="Sort column for temporal_collapse strategy.",
    )
    dedup_sort_ascending: bool | None = Field(
        default=None,
        description=(
            "Sort direction for temporal_collapse. True = ascending (earliest first), "
            "False = descending (latest first). Pair with dedup_keep='first' per contract docs."
        ),
    )
    dedup_keep: Literal["first", "last"] | None = Field(
        default=None,
        description="Which row to keep after sort. Never 'any_row' — that is a 2a concept.",
    )
    hook_spec: HookSpec | None = Field(
        default=None,
        description=(
            "Populated on GENERATE_HOOK reentry after hook generation call. "
            "Presence signals resolver to set dedup_method='hook_required' on DedupPolicy."
        ),
    )

    @model_validator(mode="after")
    def temporal_collapse_requires_sort(self) -> "GrainResolution":
        if self.dedup_strategy == "temporal_collapse":
            if self.dedup_sort_by is None:
                raise ValueError(
                    "dedup_strategy='temporal_collapse' requires dedup_sort_by to be set."
                )
            if self.dedup_sort_ascending is None:
                raise ValueError(
                    "dedup_strategy='temporal_collapse' requires dedup_sort_ascending to be set "
                    "(True for earliest, False for latest)."
                )
        if self.dedup_sort_ascending is not None and self.dedup_sort_by is None:
            raise ValueError("dedup_sort_ascending requires dedup_sort_by to be set.")
        return self

    @model_validator(mode="after")
    def hook_spec_excludes_strategy(self) -> "GrainResolution":
        if self.hook_spec is not None and self.dedup_strategy is not None:
            raise ValueError(
                "hook_spec and dedup_strategy are mutually exclusive — "
                "hook_spec resolution replaces strategy-based resolution. "
                "Execution layer infers hook path from hook_spec presence."
            )
        return self


# ---------------------------------------------------------------------------
# GrainAmbiguityHITLContext — structured identity_grain reviewer evidence
# ---------------------------------------------------------------------------


class GrainCandidateKeyEntry(BaseModel):
    """One candidate key in reviewer-ranked order (IdentityAgent: semantic plausibility)."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(
        ...,
        ge=1,
        description=(
            "1 = contract grain for this response (must match post_clean_primary_key columns); "
            "2+ = alternative override keys. In IdentityAgent output, ordering is semantic, "
            "not the key profiler's uniqueness-first rank."
        ),
    )
    columns: list[str] = Field(..., min_length=1)
    uniqueness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated key uniqueness on the profiled sample.",
    )
    notes: str | None = Field(
        default=None,
        description="Short human-readable caveat (e.g. measure columns mixed into key).",
    )


class GrainAmbiguityHITLContext(BaseModel):
    """
    Structured alternative to a freeform ``hitl_context`` string for ``identity_grain``.

    Use when the pipeline has ranked candidate keys and duplicate-group variance summaries.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_keys: list[GrainCandidateKeyEntry] = Field(
        ...,
        min_length=1,
        description=(
            "Rank 1 = same columns as post_clean_primary_key in the IdentityAgent response; "
            "later ranks = alternative grains (option candidate_key_override). uniqueness_score "
            "is profiling evidence, not the sort key for this list."
        ),
    )
    variance_profile: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Column name → human-readable variance within duplicate groups "
            "(e.g. '25%–58.8% within groups')."
        ),
    )


# ---------------------------------------------------------------------------
# TermResolution — Pass 2 output mutations
# ---------------------------------------------------------------------------


class TermResolution(BaseModel):
    """
    Mutations applied to term_config when a Pass 2 HITL item is resolved.

    When hook_spec is present, :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.resolve_items`
    or :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.apply_hook_spec` sets
    ``term_extraction='hook_required'`` and writes ``hook_spec``. Split ``year_col``/``season_col``
    are cleared when ``term_col`` is present (see resolver).

    All fields optional — resolver applies only those present.
    """

    exclude_tokens: list[str] | None = Field(
        default=None,
        description=(
            "Token prefixes to exclude from term ordering e.g. ['Custom label']. "
            "Resolver matches by prefix strip — enumerating every variant is not required."
        ),
    )
    season_map_append: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "New raw → canonical entries to append to season_map. "
            "Each dict is validated as SeasonMapEntry (canonical must be FALL|SPRING|SUMMER|WINTER)."
        ),
    )
    season_map_replace: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Full raw → canonical season map for hook-required resolutions. "
            "Raw keys must match every possible season_extractor return value when hook_spec is set."
        ),
    )
    term_col_override: str | None = Field(
        default=None,
        description=(
            "Column name for combined term encoding. Clears year_col and season_col when set."
        ),
    )
    hook_spec: HookSpec | None = Field(
        default=None,
        description=(
            "May be set after GENERATE_HOOK hook generation, or on a terminal resolution option. "
            "Resolver sets term_extraction='hook_required' and clears split columns when term_col is set."
        ),
    )

    @model_validator(mode="after")
    def hook_spec_requires_season_map_replace(self) -> "TermResolution":
        if self.hook_spec is not None and self.season_map_replace is None:
            raise ValueError(
                "season_map_replace is required when hook_spec is present — "
                "season_extractor returns raw tokens that must be mapped via season_map_replace. "
                "Provide season_map_replace with one entry per possible season_extractor return value."
            )
        return self


# ---------------------------------------------------------------------------
# HITLOption
# ---------------------------------------------------------------------------


class HITLOption(BaseModel):
    """
    One reviewer-selectable resolution for a HITL flag.

    Rules enforced by HITLItem validator:
    - 2–5 options per item.
    - Last option must always be option_id='custom'.
    - ``custom`` normally uses ``resolution=null``; when ``reentry`` is ``generate_hook``,
      ``custom`` may instead carry a **partial** resolution (e.g. ``season_map_replace`` only)
      so the resolver can persist the season map while ``reviewer_note`` drives hook code.
      Partial resolutions must **not** include ``hook_spec`` (that comes from hook generation).
    - Non-custom options must have a non-null resolution unless reentry is
      ``generate_hook`` (resolution may be null until hook generation fills hook_spec).
    """

    option_id: str = Field(
        ...,
        description="Snake-case identifier, unique within the item.",
    )
    label: str = Field(
        ...,
        description="Short display label for reviewer (~4 words).",
    )
    description: str = Field(
        ...,
        description="One sentence explaining the consequence of this choice.",
    )
    resolution: dict | None = Field(
        ...,
        description=(
            "Mutation applied by resolver on selection. Null for option_id='custom' unless "
            "reentry='generate_hook' and reviewer supplies a partial resolution (no hook_spec). "
            "Also null for non-custom options with reentry='generate_hook' before hook_spec exists."
        ),
    )
    reentry: ReentryDepth

    @model_validator(mode="after")
    def validate_resolution(self) -> "HITLOption":
        if self.option_id == "custom" and self.resolution is not None:
            if self.reentry != ReentryDepth.GENERATE_HOOK:
                raise ValueError(
                    "option_id='custom' must have resolution=null unless reentry is "
                    f"'{ReentryDepth.GENERATE_HOOK.value}' "
                    "(then optional partial resolution without hook_spec, e.g. season_map_replace)."
                )
            hs: object | None
            if isinstance(self.resolution, dict):
                hs = self.resolution.get("hook_spec")
            else:
                hs = getattr(self.resolution, "hook_spec", None)
            if hs:
                raise ValueError(
                    "option_id='custom' cannot include hook_spec in resolution — "
                    "use reviewer_note + hook generation for hook_spec."
                )
        if self.option_id != "custom" and self.resolution is None:
            if self.reentry != ReentryDepth.GENERATE_HOOK:
                raise ValueError(
                    f"Non-custom option '{self.option_id}' must have a non-null resolution "
                    f"unless reentry is '{ReentryDepth.GENERATE_HOOK.value}'."
                )
        return self


# ---------------------------------------------------------------------------
# HITLTarget
# ---------------------------------------------------------------------------


class HITLTarget(BaseModel):
    """Points the resolver at the exact config object to mutate."""

    institution_id: str
    table: str
    config: str = Field(
        ...,
        description="Which config to mutate e.g. 'grain_contract', 'term_config'.",
    )
    field: str = Field(
        ...,
        description="Which field within that config e.g. 'dedup_policy', 'season_map'.",
    )


# ---------------------------------------------------------------------------
# HITLItem
# ---------------------------------------------------------------------------


class HITLItem(BaseModel):
    """
    A single reviewable flag emitted by IdentityAgent.

    One HITLItem per ambiguity per table. A single table may emit multiple
    items if independent questions arise (e.g. grain ambiguity + dedup policy).

    Reviewer action: set ``choice`` to the 1-based index of the selected option, save file.
    hitl_resolver.py does the rest.
    """

    item_id: str
    institution_id: str
    table: str
    domain: HITLDomain
    hook_group_id: str | None = Field(
        default=None,
        description=(
            "Shared identifier for items that use the same hook. "
            "get_hook_items() returns one representative per group. "
            "apply_hook_spec() fans the generated HookSpec out to all group members "
            "when apply_to_group=True. "
            "e.g. 'shared_term_encoding_a' for three tables sharing the same term encoding."
        ),
    )
    hook_group_tables: list[str] | None = Field(
        default=None,
        description=(
            "When hook_group_id is set (typically identity_term), list every logical dataset name "
            "that shares this hook — the same names as keys under identity_term_output.json "
            "`datasets`. apply_hook_spec(apply_to_group=True) writes hook_spec to each listed "
            "table (union with HITL rows that share hook_group_id). "
            "Null or empty means only tables that have a HITLItem in this group receive updates."
        ),
    )

    hitl_question: str = Field(
        ...,
        description="Specific, actionable question naming the column, values, and decision needed.",
    )
    hitl_context: str | GrainAmbiguityHITLContext | None = Field(
        default=None,
        description=(
            "Evidence for the reviewer: either a short freeform string, or structured "
            ":class:`GrainAmbiguityHITLContext` (ranked candidate keys and variance_profile) "
            "for grain ambiguity."
        ),
    )

    options: list[HITLOption]
    target: HITLTarget
    choice: int | None = Field(
        default=None,
        description=(
            "1-indexed selection from options. "
            "Reviewer sets this to 1 … len(options). Resolver reads options[choice - 1]. "
            "null = not yet reviewed. Re-run resolver after changing choice to reapply."
        ),
    )
    reviewer_note: str | None = Field(
        default=None,
        description=(
            "Freetext correction or instruction from the reviewer. "
            "Filled in by the reviewer alongside setting choice when the selected option "
            "is custom (reentry='generate_hook') or when the draft resolution needs correction. "
            "For identity_term custom + generate_hook, pair this with a partial "
            "option resolution (e.g. season_map_replace only) so term_config.season_map is "
            "written before hook generation — reviewer_note alone does not apply structured fields. "
            "Passed as authoritative context to the hook generation LLM call. "
            "null = no reviewer note provided."
        ),
    )

    @field_validator("hook_group_tables", mode="before")
    @classmethod
    def _normalize_hook_group_tables(cls, v: object) -> list[str] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise TypeError("hook_group_tables must be a list of strings or null")
        out = [str(x).strip() for x in v if str(x).strip()]
        return out or None

    @model_validator(mode="after")
    def hook_group_tables_requires_group_id(self) -> "HITLItem":
        if self.hook_group_tables and not self.hook_group_id:
            raise ValueError("hook_group_tables requires hook_group_id to be set")
        return self

    @model_validator(mode="after")
    def validate_options(self) -> "HITLItem":
        n = len(self.options)
        if n < 2 or n > 5:
            raise ValueError(f"HITLItem must have 2–5 options, got {n}.")
        if self.options[-1].option_id != "custom":
            raise ValueError(
                "Last option must always be option_id='custom' as an escape hatch."
            )
        return self

    @model_validator(mode="after")
    def choice_in_range(self) -> "HITLItem":
        if self.choice is not None and not (1 <= self.choice <= len(self.options)):
            raise ValueError(
                f"choice={self.choice} is out of range — "
                f"must be between 1 and {len(self.options)}."
            )
        return self

    @model_validator(mode="after")
    def domain_matches_resolution_type(self) -> "HITLItem":
        for opt in self.options:
            if opt.resolution is None:
                continue
            if self.domain == HITLDomain.IDENTITY_GRAIN:
                GrainResolution.model_validate(opt.resolution)
            elif self.domain == HITLDomain.IDENTITY_TERM:
                TermResolution.model_validate(opt.resolution)
        return self

    def selected_option(self) -> HITLOption | None:
        """
        Returns the option selected by the reviewer via the choice field.
        Returns None if choice is not yet set (not yet reviewed).
        Idempotent — safe to call multiple times, always reflects current choice.
        """
        if self.choice is not None:
            return self.options[self.choice - 1]
        return None


# ---------------------------------------------------------------------------
# InstitutionHITLItems — top-level file written by IdentityAgent
# ---------------------------------------------------------------------------


class InstitutionHITLItems(BaseModel):
    """
    All HITL items for one institution run, written alongside agent output files.

    Grain inference writes: identity_grain_hitl.json
    Term normalization writes: identity_term_hitl.json

    Empty items list means no flags were raised — gate check passes immediately.
    """

    institution_id: str
    domain: Literal["grain", "term"]
    items: list[HITLItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def items_match_institution(self) -> "InstitutionHITLItems":
        for item in self.items:
            if item.institution_id != self.institution_id:
                raise ValueError(
                    f"HITLItem '{item.item_id}' institution_id '{item.institution_id}' "
                    f"does not match envelope institution_id '{self.institution_id}'."
                )
        return self

    @model_validator(mode="after")
    def items_match_domain(self) -> "InstitutionHITLItems":
        expected = (
            HITLDomain.IDENTITY_GRAIN
            if self.domain == "grain"
            else HITLDomain.IDENTITY_TERM
        )
        for item in self.items:
            if item.domain != expected:
                raise ValueError(
                    f"HITLItem '{item.item_id}' domain '{item.domain.value}' "
                    f"does not match envelope domain '{self.domain}'."
                )
        return self

    @property
    def pending(self) -> list[HITLItem]:
        """Items not yet reviewed — choice is null."""
        return [i for i in self.items if i.choice is None]

    @property
    def is_clear(self) -> bool:
        """True when all items have a choice set — gate check passes."""
        return all(i.choice is not None for i in self.items)


def get_grain_hitl_item_schema_context() -> str:
    """
    Python source for grain-pass HITL prompts only.

    Includes :class:`GrainResolution` and shared HITL shapes; omits :class:`TermResolution`
    so the model is not shown term-only fields alongside grain options.
    """
    return concat_model_sources(
        (
            HITLDomain,
            ReentryDepth,
            HookFunctionSpec,
            HookSpec,
            GrainResolution,
            GrainCandidateKeyEntry,
            GrainAmbiguityHITLContext,
            HITLOption,
            HITLTarget,
            HITLItem,
            InstitutionHITLItems,
        )
    )


def get_term_hitl_item_schema_context() -> str:
    """
    Python source for term-pass HITL prompts only.

    Includes :class:`TermResolution` and shared HITL shapes; omits :class:`GrainResolution`
    so the model is not shown dedup/grain fields alongside term options.
    """
    return concat_model_sources(
        (
            HITLDomain,
            ReentryDepth,
            HookFunctionSpec,
            HookSpec,
            TermResolution,
            HITLOption,
            HITLTarget,
            HITLItem,
            InstitutionHITLItems,
        )
    )


# Run log models — shared with SMA HITL (:mod:`edvise.genai.mapping.shared.hitl.run_log`).
