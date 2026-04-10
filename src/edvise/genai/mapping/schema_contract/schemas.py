"""
**Enriched** schema-contract Pydantic models (think of this module as ``enriched_schemas``).

Despite the filename ``schemas.py``, everything here describes the **enriched** institution
JSON IdentityAgent produces **after** freezing per-dataset schemas: per-dataset
``training`` metadata (column stats, samples, etc.) plus envelope fields such as
``school_id`` / ``school_name``. That is **not** the same artifact as the raw multi-dataset
dict from :func:`~edvise.data_audit.custom_cleaning.build_schema_contract` or
:func:`~edvise.genai.mapping.schema_contract.build_from_school_config.build_schema_contract_from_config`,
which only freeze dtypes / keys / hashes from cleaned frames — enrichment is applied in
:mod:`edvise.genai.mapping.identity_agent.execution.contract_builder`.

IdentityAgent writes this shape (e.g. :func:`~edvise.genai.mapping.identity_agent.execution.contract_builder.save_enriched_schema_contract`);
Schema Mapping Agent consumes it (e.g. :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.prompt_builder.summarize_schema_contract`).
Files are typically named ``{school_id}_schema_contract.json``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    Single schema contract document for an institution — SMA prompt/eval input shape.

    Validated JSON matches files written by IdentityAgent enrichment and summarized by
    :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.prompt_builder.summarize_schema_contract`.
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
    """Parse a loaded schema-contract dict (e.g. from JSON) into the canonical model."""
    return EnrichedSchemaContractForSMA.model_validate(data)


__all__ = [
    "EnrichedSchemaContractForSMA",
    "FrozenDatasetSchemaForSMA",
    "SchemaContractColumnDetail",
    "SchemaContractTrainingBlock",
    "parse_enriched_schema_contract_for_sma",
]
