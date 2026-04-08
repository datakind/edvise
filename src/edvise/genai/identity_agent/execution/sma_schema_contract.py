"""
Pydantic contract for the **enriched** frozen schema JSON consumed by Schema Mapping Agent.

Built by IdentityAgent execution helpers (e.g. :func:`build_enriched_schema_contract_for_dataset`,
:func:`_build_enriched_schema_contract`). This is the single canonical shape for SMA prompts and
eval loaders that read ``{school_id}_schema_contract.json``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --- Training / enrichment (per dataset, under ``training``) ---


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


class SchemaContractTrainingBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    num_rows: int
    num_columns: int
    column_normalization: dict[str, Any]
    column_details: list[SchemaContractColumnDetail]


class FrozenDatasetSchemaForSMA(BaseModel):
    """
    One dataset entry: output of :func:`~edvise.data_audit.custom_cleaning.freeze_schema`
    plus ``training`` from IdentityAgent enrichment.
    """

    model_config = ConfigDict(extra="allow")

    normalized_columns: dict[str, str]
    dtypes: dict[str, str]
    non_null_columns: list[str]
    unique_keys: list[str]
    null_tokens: list[str]
    boolean_map: dict[str, bool]
    column_order_hash: str | None = None
    training: SchemaContractTrainingBlock


class EnrichedSchemaContractForSMA(BaseModel):
    """
    Single schema contract document for an institution — **this is what SMA consumes**.

    Validated JSON matches files written by :func:`save_enriched_schema_contract` and expected by
    :func:`~edvise.genai.schema_mapping_agent.manifest.prompt_builder.summarize_schema_contract`.
    """

    model_config = ConfigDict(extra="forbid")

    created_at: str | None = None
    null_tokens: list[str] = Field(default_factory=list)
    school_id: str
    school_name: str
    notes: str | None = None
    student_id_alias: str | None = None
    datasets: dict[str, FrozenDatasetSchemaForSMA]


def parse_enriched_schema_contract_for_sma(
    data: dict[str, Any],
) -> EnrichedSchemaContractForSMA:
    """Parse a loaded schema-contract dict (e.g. from JSON) into the canonical SMA model."""
    return EnrichedSchemaContractForSMA.model_validate(data)


__all__ = [
    "EnrichedSchemaContractForSMA",
    "FrozenDatasetSchemaForSMA",
    "SchemaContractColumnDetail",
    "SchemaContractTrainingBlock",
    "parse_enriched_schema_contract_for_sma",
]
