"""
Pydantic models for IdentityAgent HITL items.

HITLItem is the unit of human review — one item per ambiguity, per table.
Reviewer deletes 2 of 3 options, leaving exactly one, then sets status to 'resolved'.
hitl_resolver.py reads these files and applies the selected resolution to the
relevant config (grain_contract or term_config).

Scope: IdentityAgent Pass 1 (grain) and Pass 2 (term) only for M2.
SCHEMA_MAPPING and TRANSFORM domains are stubs for future use.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

from edvise.genai.identity_agent.grain_inference.schemas import HookSpec


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HITLDomain(str, Enum):
    IDENTITY_GRAIN = "identity_grain"
    IDENTITY_TERM  = "identity_term"
    SCHEMA_MAPPING = "schema_mapping"  # future
    TRANSFORM      = "transform"       # future


class ReentryDepth(str, Enum):
    TERMINAL      = "terminal"       # apply resolution, continue to next stage
    GENERATE_HOOK = "generate_hook"  # trigger hook gen LLM call, then continue


class HITLStatus(str, Enum):
    PENDING  = "pending"
    RESOLVED = "resolved"
    SKIPPED  = "skipped"


# ---------------------------------------------------------------------------
# GrainResolution — Pass 1 output mutations
# ---------------------------------------------------------------------------

class GrainResolution(BaseModel):
    """
    Mutations applied to the grain contract when a Pass 1 HITL item is resolved.

    dedup_strategy excludes 'policy_required' — that is the current state that
    triggered the HITL item, never a valid resolution target.

    When hook_spec is present, resolver sets DedupPolicy.dedup_method='hook_required'
    and DedupPolicy.hook_spec from this value. When hook_spec is absent, resolver
    sets DedupPolicy.dedup_method='standard'.

    All fields optional — resolver applies only those present.
    """
    candidate_key_override: list[str] | None = Field(
        default=None,
        description="Reviewer-corrected columns forming the post-clean primary key.",
    )
    dedup_strategy: Literal[
        "true_duplicate",
        "temporal_collapse",
        "no_dedup",
    ] | None = Field(
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
        if self.dedup_strategy == "temporal_collapse" and self.dedup_sort_by is None:
            raise ValueError(
                "dedup_strategy='temporal_collapse' requires dedup_sort_by to be set."
            )
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
# TermResolution — Pass 2 output mutations
# ---------------------------------------------------------------------------

class TermResolution(BaseModel):
    """
    Mutations applied to term_config when a Pass 2 HITL item is resolved.

    When hook_spec is present, resolver sets TermOrderConfig.term_extraction='hook_required'
    and TermOrderConfig.hook_spec from this value.

    All fields optional — resolver applies only those present.
    """
    exclude_tokens: list[str] | None = Field(
        default=None,
        description=(
            "Token prefixes to exclude from term ordering e.g. ['Med Year']. "
            "Resolver matches by prefix strip — enumerating every variant is not required."
        ),
    )
    season_map_append: list[dict[str, str]] | None = Field(
        default=None,
        description="New raw → canonical entries to append to season_map.",
    )
    term_col_override: str | None = Field(
        default=None,
        description="Column name to use instead of LLM-selected term_col.",
    )
    hook_spec: HookSpec | None = Field(
        default=None,
        description=(
            "Populated on GENERATE_HOOK reentry after hook generation call. "
            "Presence signals resolver to set term_extraction='hook_required' on TermOrderConfig."
        ),
    )


# ---------------------------------------------------------------------------
# HITLOption
# ---------------------------------------------------------------------------

AnyResolution = Annotated[
    Union[GrainResolution, TermResolution],
    Field(discriminator=None),
]


class HITLOption(BaseModel):
    """
    One reviewer-selectable resolution for a HITL flag.

    Rules enforced by HITLItem validator:
    - Exactly 2–3 options per item.
    - Last option must always be option_id='custom' with resolution=None.
    - All non-custom options must have a non-null resolution.
    """
    option_id:   str = Field(
        ...,
        description="Snake-case identifier, unique within the item.",
    )
    label:       str = Field(
        ...,
        description="Short display label for reviewer (~4 words).",
    )
    description: str = Field(
        ...,
        description="One sentence explaining the consequence of this choice.",
    )
    resolution:  AnyResolution | None = Field(
        ...,
        description="Mutation applied by resolver on selection. Null only for option_id='custom'.",
    )
    reentry:     ReentryDepth

    @model_validator(mode="after")
    def validate_resolution(self) -> "HITLOption":
        if self.option_id == "custom" and self.resolution is not None:
            raise ValueError("option_id='custom' must have resolution=null.")
        if self.option_id != "custom" and self.resolution is None:
            raise ValueError(
                f"Non-custom option '{self.option_id}' must have a non-null resolution."
            )
        return self


# ---------------------------------------------------------------------------
# HITLTarget
# ---------------------------------------------------------------------------

class HITLTarget(BaseModel):
    """Points the resolver at the exact config object to mutate."""
    institution_id: str
    table:          str
    config:         str = Field(
        ...,
        description="Which config to mutate e.g. 'grain_contract', 'term_config'.",
    )
    field:          str = Field(
        ...,
        description="Which field within that config e.g. 'dedup_policy', 'season_map'.",
    )


# ---------------------------------------------------------------------------
# HITLResolution — written back by resolver after reviewer acts
# ---------------------------------------------------------------------------

class HITLResolution(BaseModel):
    """Populated by hitl_resolver.py after the reviewer selects an option."""
    selected_option_id: str
    custom_instruction: str | None = None  # only when option_id='custom'
    resolved_by:        str | None = None
    resolved_at:        str | None = None  # ISO datetime string


# ---------------------------------------------------------------------------
# HITLItem
# ---------------------------------------------------------------------------

class HITLItem(BaseModel):
    """
    A single reviewable flag emitted by IdentityAgent.

    One HITLItem per ambiguity per table. A single table may emit multiple
    items if independent questions arise (e.g. grain ambiguity + dedup policy).

    Reviewer action: delete 2 of 3 options, set status='resolved', save file.
    hitl_resolver.py does the rest.
    """
    item_id:        str
    institution_id: str
    table:          str
    domain:         HITLDomain
    hook_group_id:  str | None = Field(
        default=None,
        description=(
            "Shared identifier for items that use the same hook. "
            "get_hook_items() returns one representative per group. "
            "apply_hook_spec() fans the generated HookSpec out to all group members "
            "when apply_to_group=True. "
            "e.g. 'jjc_term_format_a' for three tables sharing the same term encoding."
        ),
    )

    hitl_question: str = Field(
        ...,
        description="Specific, actionable question naming the column, values, and decision needed.",
    )
    hitl_context:  str | None = Field(
        default=None,
        description=(
            "Raw values or samples the LLM was looking at when it raised the flag. "
            "Surfaces evidence to the reviewer without them digging into the data."
        ),
    )

    options:    list[HITLOption]
    target:     HITLTarget

    status:     HITLStatus     = HITLStatus.PENDING
    resolution: HITLResolution | None = None

    @model_validator(mode="after")
    def validate_options(self) -> "HITLItem":
        n = len(self.options)
        if n < 2 or n > 3:
            raise ValueError(f"HITLItem must have 2–3 options, got {n}.")
        if self.options[-1].option_id != "custom":
            raise ValueError(
                "Last option must always be option_id='custom' as an escape hatch."
            )
        return self

    @model_validator(mode="after")
    def resolved_requires_resolution(self) -> "HITLItem":
        if self.status == HITLStatus.RESOLVED and self.resolution is None:
            raise ValueError("status='resolved' requires a populated resolution.")
        return self

    @model_validator(mode="after")
    def domain_matches_resolution_type(self) -> "HITLItem":
        for opt in self.options:
            if opt.resolution is None:
                continue
            if self.domain == HITLDomain.IDENTITY_GRAIN and not isinstance(opt.resolution, GrainResolution):
                raise ValueError(
                    f"domain='identity_grain' requires GrainResolution, "
                    f"got {type(opt.resolution).__name__} on option '{opt.option_id}'."
                )
            if self.domain == HITLDomain.IDENTITY_TERM and not isinstance(opt.resolution, TermResolution):
                raise ValueError(
                    f"domain='identity_term' requires TermResolution, "
                    f"got {type(opt.resolution).__name__} on option '{opt.option_id}'."
                )
        return self

    def selected_option(self) -> HITLOption | None:
        """
        Returns the single remaining option after reviewer deleted the others.
        Returns None if still pending (more than one option present).
        """
        if len(self.options) == 1:
            return self.options[0]
        return None


# ---------------------------------------------------------------------------
# InstitutionHITLItems — top-level file written by IdentityAgent
# ---------------------------------------------------------------------------

class InstitutionHITLItems(BaseModel):
    """
    All HITL items for one institution run, written alongside agent output files.

    Pass 1 writes: identity_pass1_hitl.json
    Pass 2 writes: identity_pass2_hitl.json

    Empty items list means no flags were raised — gate check passes immediately.
    """
    institution_id: str
    pass_number:    Literal[1, 2]
    items:          list[HITLItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def items_match_institution(self) -> "InstitutionHITLItems":
        for item in self.items:
            if item.institution_id != self.institution_id:
                raise ValueError(
                    f"HITLItem '{item.item_id}' institution_id '{item.institution_id}' "
                    f"does not match envelope institution_id '{self.institution_id}'."
                )
        return self

    @property
    def pending(self) -> list[HITLItem]:
        return [i for i in self.items if i.status == HITLStatus.PENDING]

    @property
    def is_clear(self) -> bool:
        """True when no items are pending — gate check passes."""
        return len(self.pending) == 0