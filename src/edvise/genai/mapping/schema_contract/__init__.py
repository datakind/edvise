"""Public exports for :mod:`edvise.genai.mapping.schema_contract`.

- **Enriched schema-contract models** — :mod:`edvise.genai.mapping.schema_contract.schemas`
  (despite the name, that module is the **enriched** JSON shape; see its module docstring).
- **Load / ``clean_dataset`` / freeze** (raw contract dict, not enriched) —
  :mod:`edvise.genai.mapping.schema_contract.build_from_school_config`, so importing only
  Pydantic models from the package root stays lightweight.
"""

from __future__ import annotations

from .schemas import (
    EnrichedSchemaContractForSMA,
    FrozenDatasetSchemaForSMA,
    SchemaContractColumnDetail,
    SchemaContractTrainingBlock,
    parse_enriched_schema_contract_for_sma,
)

__all__ = [
    "EnrichedSchemaContractForSMA",
    "FrozenDatasetSchemaForSMA",
    "SchemaContractColumnDetail",
    "SchemaContractTrainingBlock",
    "parse_enriched_schema_contract_for_sma",
]
