"""Pydantic models for IdentityAgent grain contract output (LLM-validated JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.shared.hitl.hook_spec.schemas import HookFunctionSpec, HookSpec
from edvise.genai.mapping.identity_agent.utilities import (
    concat_model_sources,
    get_top_level_assign_source,
)
from edvise.utils.data_cleaning import convert_to_snake_case

# Valid `dedup_policy.strategy` values (JSON must use these exact strings).
DedupStrategy = Literal[
    "true_duplicate",
    "temporal_collapse",
    "categorical_priority",
    "suffix_identifier",
    "no_dedup",
    "policy_required",  # current state only — never a valid resolution target
]


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: DedupStrategy
    sort_by: str | None = None
    sort_ascending: bool | None = Field(
        default=None,
        description=(
            "Sort direction for temporal_collapse. "
            "True = ascending (earliest first), False = descending (latest first). "
            "Always pair with keep='first'. Never use keep='last'."
        ),
    )
    keep: Literal["first", "last"] | None = None
    suffix_column: str | None = Field(
        default=None,
        description=(
            "When strategy is suffix_identifier: a grain column (must appear in post_clean_primary_key) "
            "whose values are suffixed with -1, -2, ... within each grain key group so all rows are kept. "
            "Required for that strategy; never use a non-grain column even if more readable."
        ),
    )
    priority_column: str | None = Field(
        default=None,
        description=(
            "When strategy is categorical_priority: column whose categorical values are ranked. "
            "Required for that strategy; must be null for other strategies."
        ),
    )
    priority_order: list[str] | None = Field(
        default=None,
        description=(
            "When strategy is categorical_priority: explicit value order, highest priority first. "
            "The executor first tries exact value equality, then match-by-substring: a cell's "
            "string may contain a listed token (e.g. B.S. in 'Psychology, B.S.'). If multiple "
            "listed tokens are substrings, the longest token wins, then the earlier list position. "
            "Unmatched values are ranked last. Required (non-empty) for that strategy."
        ),
    )
    notes: str = ""
    hook_spec: HookSpec | None = Field(
        default=None,
        description=(
            "Populated when strategy='policy_required' and a custom hook is needed. "
            "Null at flag time — written by resolver after hook generation call. "
            "Execution layer infers hook path from hook_spec presence; "
            "no separate dedup_method field is needed."
        ),
    )

    @model_validator(mode="after")
    def hook_spec_requires_policy_required(self) -> "DedupPolicy":
        if self.hook_spec is not None and self.strategy != "policy_required":
            raise ValueError(
                "hook_spec should only be populated when strategy='policy_required'."
            )
        return self

    @model_validator(mode="after")
    def sort_ascending_requires_sort_by(self) -> "DedupPolicy":
        if self.sort_ascending is not None and self.sort_by is None:
            raise ValueError("sort_ascending requires sort_by to be set.")
        if self.strategy == "temporal_collapse" and self.sort_ascending is None:
            raise ValueError(
                "temporal_collapse requires sort_ascending to be explicitly set — "
                "True for earliest, False for latest."
            )
        return self

    @model_validator(mode="after")
    def suffix_identifier_requires_suffix_column(self) -> "DedupPolicy":
        if self.strategy == "suffix_identifier" and not (
            self.suffix_column and str(self.suffix_column).strip()
        ):
            raise ValueError(
                "dedup strategy suffix_identifier requires suffix_column to be a non-empty column name."
            )
        return self

    @model_validator(mode="after")
    def strategy_specific_field_consistency(self) -> "DedupPolicy":
        """Enforce that optional dedup fields are only set for the strategies that use them."""
        s = self.strategy
        if s == "categorical_priority":
            if not self.priority_column or not str(self.priority_column).strip():
                raise ValueError(
                    "categorical_priority requires a non-empty priority_column."
                )
            if self.priority_order is None or not len(self.priority_order):
                raise ValueError(
                    "categorical_priority requires priority_order to be a non-null non-empty list."
                )
            for name, v in (
                ("sort_by", self.sort_by),
                ("keep", self.keep),
                ("sort_ascending", self.sort_ascending),
                ("suffix_column", self.suffix_column),
            ):
                if v is not None:
                    raise ValueError(
                        f"categorical_priority requires {name} to be null; use "
                        f"priority_column and priority_order only."
                    )
            return self

        if self.priority_column is not None or self.priority_order is not None:
            raise ValueError(
                f"dedup strategy {s!r} may not set priority_column or priority_order; "
                "those fields are only for strategy=categorical_priority."
            )
        if s not in ("suffix_identifier",) and self.suffix_column is not None:
            raise ValueError(
                f"dedup strategy {s!r} requires suffix_column to be null; "
                "it is only used for strategy=suffix_identifier."
            )
        if s in ("true_duplicate", "no_dedup") and (
            self.sort_by is not None
            or self.keep is not None
            or self.sort_ascending is not None
        ):
            raise ValueError(
                f"dedup strategy {s!r} requires sort_by, keep, and sort_ascending to be null."
            )
        if s == "suffix_identifier" and (
            self.sort_by is not None
            or self.keep is not None
            or self.sort_ascending is not None
        ):
            raise ValueError(
                "suffix_identifier requires sort_by, keep, and sort_ascending to be null."
            )
        return self


class GrainContract(BaseModel):
    """
    Grain contract for one institution dataset from IdentityAgent **grain** stage (grain only).

    Term column config is produced in the **term** stage as :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermContract`.

    When ``learner_id_alias`` is set, ``post_clean_primary_key`` and ``join_keys_for_2a``
    should name that column **as in the pre-canonical-rename frame** (the same string as
    ``learner_id_alias``), so dedup/join keys stay consistent; downstream cleaning renames
    it to canonical ``student_id`` for the frozen schema contract (GenAI uses learner naming
    to align with Schema Mapping Agent ``learner_id`` migration).
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    institution_id: str
    table: str
    learner_id_alias: str | None = Field(
        default=None,
        description=(
            "Institution learner/student-identifier column **as shown in the column list** "
            "(header-normalized, typically snake_case), e.g. student_id_randomized_datakind. "
            "Use null when the column is already student_id after normalization, or when this "
            "table's grain has no person identifier. Downstream cleaning maps this to canonical "
            "student_id once (see DatasetConfig.student_id_alias / CleaningConfig.student_id_alias)."
        ),
    )
    post_clean_primary_key: list[str] = Field(
        ...,
        description=(
            "Grain primary key column names aligned with the frame used for grain dedup: when "
            "learner_id_alias is set, list that column name (not literal student_id) wherever the "
            "learner identifier is part of the key. Maps to schema contract unique_keys after the "
            "canonical student_id rename."
        ),
    )
    dedup_policy: DedupPolicy
    row_selection_required: bool
    join_keys_for_2a: list[str] = Field(
        ...,
        description=(
            "Join keys for SchemaMappingAgent 2a; same naming convention as post_clean_primary_key "
            "for the learner identifier column when learner_id_alias is set."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Agent confidence in the proposed grain and dedup policy (same 0.0–1.0 scale as "
            "Schema Mapping Agent). Drives HITL — at or below the shared "
            "PIPELINE_HITL_CONFIDENCE_THRESHOLD requires hitl_flag true."
        ),
    )
    hitl_flag: bool
    reasoning: str
    notes: str = ""

    @property
    def unique_keys(self) -> list[str]:
        """Alias for ``post_clean_primary_key`` (schema contract naming)."""
        return self.post_clean_primary_key

    @model_validator(mode="before")
    @classmethod
    def _legacy_student_id_alias_field(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "student_id_alias" in out and "learner_id_alias" not in out:
            out["learner_id_alias"] = out.pop("student_id_alias", None)
        elif "student_id_alias" in out and "learner_id_alias" in out:
            out.pop("student_id_alias", None)
        return out

    @field_validator("learner_id_alias", mode="before")
    @classmethod
    def _empty_learner_id_alias_to_none(
        cls, v: object, info: ValidationInfo
    ) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            # Canonicalize to normalized column naming used by clean_dataset.
            alias_norm = convert_to_snake_case(s)
            ctx = info.context if isinstance(info.context, dict) else None
            available = (
                ctx.get("available_columns_normalized")
                if isinstance(ctx, dict)
                else None
            )
            if available is not None:
                allowed = {str(c).strip() for c in available if str(c).strip()}
                if alias_norm not in allowed:
                    sample = sorted(allowed)[:20]
                    raise ValueError(
                        "learner_id_alias must match a dataset column after normalization; "
                        f"got {s!r} (normalized: {alias_norm!r}). "
                        f"Available normalized columns sample: {sample!r}"
                    )
            return alias_norm
        return str(v)

    @model_validator(mode="after")
    def low_confidence_requires_hitl(self) -> GrainContract:
        if self.confidence <= PIPELINE_HITL_CONFIDENCE_THRESHOLD and not self.hitl_flag:
            raise ValueError(
                f"hitl_flag must be true when confidence is at or below "
                f"{PIPELINE_HITL_CONFIDENCE_THRESHOLD}"
            )
        return self


class InstitutionGrainContract(BaseModel):
    """
    Single JSON artifact for one institution: all dataset-level :class:`GrainContract` values.

    Keys in ``datasets`` match logical dataset names (same keys as ``inputs.toml`` / ``SchoolMappingConfig.datasets``).
    Use for testing or handoff instead of N separate per-table JSON files.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

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


def get_grain_contract_schema_context() -> str:
    """Python source for grain-stage contract models (per-dataset ``GrainContract`` JSON)."""
    # Module-level ``DedupStrategy`` is the Literal set for ``dedup_policy.strategy``; it is not
    # included in ``getsource`` of the model classes, so we prepend it for the system prompt.
    dedup_strategy = get_top_level_assign_source(__file__, "DedupStrategy")
    rest = concat_model_sources(
        (HookFunctionSpec, HookSpec, DedupPolicy, GrainContract)
    )
    if not dedup_strategy:
        return rest
    return f"{dedup_strategy}\n\n{rest}"
