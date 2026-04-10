"""Public exports for :mod:`edvise.genai.mapping.schema_contract`.

- **Frozen + enriched schema-contract models** — :mod:`edvise.genai.mapping.schema_contract.schemas`
  (raw ``build_schema_contract`` JSON uses ``student_id_alias``; Pydantic models map that to
  ``learner_id_alias`` for GenAI/SMA; see module docstring).
- **Load / ``clean_dataset`` / freeze** (raw contract dict, not enriched) —
  :mod:`edvise.genai.mapping.schema_contract.build_from_school_config`, so importing only
  Pydantic models from the package root stays lightweight.
"""

from __future__ import annotations

from .schemas import (
    BaseFrozenSchemaContract,
    EnrichedSchemaContractForSMA,
    FrozenDatasetSchemaCore,
    FrozenDatasetSchemaForSMA,
    SchemaContractColumnDetail,
    SchemaContractTrainingBlock,
    assert_build_schema_contract_matches_base_model,
    parse_base_frozen_schema_contract,
    parse_enriched_schema_contract_for_sma,
)

__all__ = [
    "BaseFrozenSchemaContract",
    "EnrichedSchemaContractForSMA",
    "FrozenDatasetSchemaCore",
    "FrozenDatasetSchemaForSMA",
    "SchemaContractColumnDetail",
    "SchemaContractTrainingBlock",
    "assert_build_schema_contract_matches_base_model",
    "parse_base_frozen_schema_contract",
    "parse_enriched_schema_contract_for_sma",
]
