"""Pydantic models for IdentityAgent grain contract output (LLM-validated JSON)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from edvise.genai.mapping.identity_agent.utilities import concat_model_sources

# Below this confidence score, `hitl_flag` must be true (ambiguous grain / policy required).
IDENTITY_CONFIDENCE_HITL_THRESHOLD: float = 0.5

# Valid `dedup_policy.strategy` values (JSON must use these exact strings).
DedupStrategy = Literal[
    "true_duplicate",
    "temporal_collapse",
    "no_dedup",
    "policy_required",  # current state only — never a valid resolution target
]


class HookFunctionSpec(BaseModel):
    """
    Spec for one generated hook function.

    ``draft`` is the **full function definition** (``def`` line through body) as one string.
    ``signature`` is optional human/LLM metadata; materialize does not use it.
    ``description`` documents intent for reviewers; validation is ``ast.parse`` → optional
    pyflakes → :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.validate_hook` signature check.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    signature: str | None = Field(
        default=None,
        description="Optional; not used by materialize (draft carries the full def).",
    )
    description: str
    draft: str | None = Field(
        default=None,
        description="Complete Python function definition: signature and body as one string.",
    )


class HookSpec(BaseModel):
    """
    File path + function specs for a generated hook.
    Shared shape between term extraction hooks and dedup hooks.

    ``file`` is optional in raw LLM JSON; hook generation overwrites it with a canonical path
    from ``institution_id`` and domain before persisting to config.
    """

    model_config = ConfigDict(extra="forbid")

    file: str | None = Field(
        default=None,
        description=(
            "Relative path to the materialized module; set by the pipeline, not the LLM "
            "(e.g. identity_hooks/<institution_id>/dedup_hooks.py under bronze_volumes_path)."
        ),
    )
    functions: list[HookFunctionSpec]


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
    notes: str = ""
    hook_spec: HookSpec | None = Field(
        default=None,
        description=(
            "Populated when strategy='policy_required' and a custom hook is needed. "
            "Null at flag time — written by resolver after hook generation call. "
            "Execution (apply_grain_dedup / clean_dataset dedupe_fn) loads hook_spec.file under "
            "hook_modules_root (e.g. bronze_volumes_path) and runs the dedup callable per key-group."
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
            "student_id once (see CleaningConfig.student_id_alias)."
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
            "Schema Mapping Agent). Drives HITL — scores below the documented threshold require "
            "hitl_flag true."
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
    def _empty_learner_id_alias_to_none(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return str(v).strip() or None

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
        """Return the same mapping as institution grain runners / schema merge APIs expect."""
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
    return concat_model_sources(
        (HookFunctionSpec, HookSpec, DedupPolicy, GrainContract)
    )
