"""
**Enriched** schema-contract Pydantic models (think of this module as ``enriched_schemas``).

Despite the filename ``schemas.py``, most types here describe the **enriched** institution
JSON IdentityAgent produces **after** freezing per-dataset schemas: per-dataset
``training`` metadata (column stats, samples, etc.) plus envelope fields such as
``school_id`` / ``school_name``.

**Base frozen contract (GenAI view)** — :class:`BaseFrozenSchemaContract` / :class:`FrozenDatasetSchemaCore`
validate the same structural dict as :func:`~edvise.data_audit.custom_cleaning.build_schema_contract`,
but the **envelope alias field is renamed**: data audit and ``build_schema_contract`` JSON use
``student_id_alias`` (see :class:`~edvise.data_audit.custom_cleaning.SchemaContractMeta`); these
models expose ``learner_id_alias`` for Schema Mapping Agent / learner-oriented naming. Parsing
maps ``student_id_alias`` → ``learner_id_alias`` (see :class:`BaseFrozenSchemaContract` validator).

Enrichment is applied in :mod:`edvise.genai.mapping.identity_agent.execution.contract_builder`.
IdentityAgent writes :class:`EnrichedSchemaContractForSMA`; Schema Mapping Agent consumes it.
Files are typically named ``{school_id}_schema_contract.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchemaContractColumnDetail(BaseModel):
    """One column row in ``training.column_details`` (SMA ``summarize_schema_contract`` input)."""

    model_config = ConfigDict(extra="forbid")

    original_name: str
    normalized_name: str
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: list[str] = Field(default_factory=list)
    unique_values: list[str] | None = None
    inferred_dtype: str | None = Field(
        default=None,
        description="Legacy only; prefer frozen dtypes on the dataset when present.",
    )


class TermNormalizationSummary(BaseModel):
    """
    How IdentityAgent derived Edvise term columns for this dataset — source column(s) after the same
    snake_case normalization as ``clean_dataset``, and which column was wired as ``CleanSpec.term_column``.

    SMA uses this for mapping context (e.g. cohort entry_term vs raw institutional codes).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["single_column", "year_season_columns"]
    term_extraction: Literal["standard", "hook_required"]
    term_col: str | None = Field(
        default=None,
        description="Single source column when mode is single_column (normalized name).",
    )
    year_col: str | None = Field(
        default=None,
        description="Year column when mode is year_season_columns (normalized name).",
    )
    season_col: str | None = Field(
        default=None,
        description="Season column when mode is year_season_columns (normalized name).",
    )
    clean_spec_term_column: str = Field(
        description=(
            "Column name passed to CleanSpec.term_column — same as "
            "term_order_column_for_clean_dataset (year column for split configs)."
        )
    )

    @model_validator(mode="after")
    def _columns_match_mode(self) -> TermNormalizationSummary:
        if self.mode == "single_column":
            if not self.term_col:
                raise ValueError("single_column requires term_col")
            if self.year_col is not None or self.season_col is not None:
                raise ValueError("single_column must not set year_col or season_col")
        else:
            if not self.year_col or not self.season_col:
                raise ValueError("year_season_columns requires year_col and season_col")
            if self.term_col is not None:
                raise ValueError("year_season_columns must not set term_col")
        return self


class SchemaContractTrainingBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    num_rows: int
    num_columns: int
    column_normalization: dict[str, Any]
    column_details: list[SchemaContractColumnDetail]
    term_normalization: TermNormalizationSummary | None = Field(
        default=None,
        description=(
            "IdentityAgent term source columns (normalized) when a TermOrderConfig was supplied "
            "for enriched contract build; omit or null when term normalization was not configured."
        ),
    )


class FrozenDatasetSchemaCore(BaseModel):
    """
    One dataset entry from :func:`~edvise.data_audit.custom_cleaning.freeze_schema`
    (no ``training`` block).
    """

    model_config = ConfigDict(extra="allow")

    normalized_columns: dict[str, str]
    dtypes: dict[str, str]
    non_null_columns: list[str]
    unique_keys: list[str]
    null_tokens: list[str]
    boolean_map: dict[str, bool]
    column_order_hash: str | None = None


class BaseFrozenSchemaContract(BaseModel):
    """
    GenAI frozen schema contract envelope — per-dataset shape matches
    :func:`~edvise.data_audit.custom_cleaning.build_schema_contract`, but the raw dict's
    ``student_id_alias`` key is represented here as ``learner_id_alias`` for SMA / learner naming.

    Raw ``build_schema_contract`` output uses ``student_id_alias`` (data audit convention);
    :meth:`model_validate` accepts that dict and maps the field onto ``learner_id_alias``.
    """

    model_config = ConfigDict(extra="forbid")

    created_at: str | None = None
    null_tokens: list[str] = Field(default_factory=list)
    learner_id_alias: str | None = None
    datasets: Mapping[str, FrozenDatasetSchemaCore]

    @model_validator(mode="before")
    @classmethod
    def _data_audit_student_id_alias_to_learner_id_alias(cls, data: Any) -> Any:
        """Map ``student_id_alias`` (custom_cleaning / build_schema_contract) → ``learner_id_alias``."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "student_id_alias" in out:
            if out.get("learner_id_alias") is None:
                out["learner_id_alias"] = out.get("student_id_alias")
            out.pop("student_id_alias", None)
        return out


class FrozenDatasetSchemaForSMA(FrozenDatasetSchemaCore):
    """
    One dataset entry: output of :func:`~edvise.data_audit.custom_cleaning.freeze_schema`
    plus ``training`` from IdentityAgent enrichment.
    """

    training: SchemaContractTrainingBlock


class EnrichedSchemaContractForSMA(BaseFrozenSchemaContract):
    """
    Single schema contract document for an institution — SMA prompt/eval input shape.

    Extends :class:`BaseFrozenSchemaContract` with ``school_id`` / ``school_name`` / ``notes``
    and per-dataset :class:`FrozenDatasetSchemaForSMA` (includes ``training``).

    ``canonical_learner_column`` is copied from the frozen base contract by IdentityAgent
    (:func:`~edvise.genai.mapping.identity_agent.execution.contract_builder.build_enriched_schema_contract_for_institution`)
    when present; it names the person-key column after cleaning (``student_id`` vs ``learner_id``).
    """

    model_config = ConfigDict(extra="forbid")

    school_id: str
    school_name: str
    notes: str | None = None
    canonical_learner_column: Literal["student_id", "learner_id"] | None = None
    datasets: Mapping[str, FrozenDatasetSchemaForSMA]


def parse_base_frozen_schema_contract(data: dict[str, Any]) -> BaseFrozenSchemaContract:
    """Parse a raw schema-contract dict from :func:`~edvise.data_audit.custom_cleaning.build_schema_contract`."""
    return BaseFrozenSchemaContract.model_validate(data)


def parse_enriched_schema_contract_for_sma(
    data: dict[str, Any],
) -> EnrichedSchemaContractForSMA:
    """Parse a loaded schema-contract dict (e.g. from JSON) into the canonical model."""
    return EnrichedSchemaContractForSMA.model_validate(data)


def assert_build_schema_contract_matches_base_model(
    cleaned_map: dict[str, Any],
    specs: dict[str, Any],
    *,
    meta: Any = None,
) -> BaseFrozenSchemaContract:
    """
    Build via :func:`~edvise.data_audit.custom_cleaning.build_schema_contract` and assert the
    result validates as :class:`BaseFrozenSchemaContract` (runtime alignment check).
    """
    from edvise.data_audit.custom_cleaning import build_schema_contract

    raw = build_schema_contract(cleaned_map, specs, meta=meta)
    return parse_base_frozen_schema_contract(raw)


__all__ = [
    "BaseFrozenSchemaContract",
    "EnrichedSchemaContractForSMA",
    "FrozenDatasetSchemaCore",
    "FrozenDatasetSchemaForSMA",
    "SchemaContractColumnDetail",
    "SchemaContractTrainingBlock",
    "TermNormalizationSummary",
    "assert_build_schema_contract_matches_base_model",
    "parse_base_frozen_schema_contract",
    "parse_enriched_schema_contract_for_sma",
]
