"""
Backward-compatible import path for schema contract training helpers.

Prefer :mod:`edvise.genai.identity_agent.execution.contract_builder`.
"""

from edvise.genai.identity_agent.execution.contract_builder import (
    UNIQUE_VALUES_MAX_CARDINALITY,
    build_enriched_schema_contract_for_dataset,
    build_training_example_from_schema_contract,
    process_all_schools,
    process_school_dataset,
    save_enriched_schema_contract,
    save_enriched_schema_contracts,
)

__all__ = [
    "UNIQUE_VALUES_MAX_CARDINALITY",
    "build_enriched_schema_contract_for_dataset",
    "build_training_example_from_schema_contract",
    "process_all_schools",
    "process_school_dataset",
    "save_enriched_schema_contract",
    "save_enriched_schema_contracts",
]
