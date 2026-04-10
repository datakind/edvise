from __future__ import annotations

from typing import Any

from .contract_utilities import (
    apply_grain_dedup,
    apply_grain_execution,
    apply_term_order_from_contract,
    build_dedupe_fn_from_grain_contract,
    canonicalize_grain_contract_learner_id_alias,
)
from .contract_builder import (
    UNIQUE_VALUES_MAX_CARDINALITY,
    build_enriched_schema_contract_for_institution,
    build_enriched_schema_contract_for_dataset,
    build_schema_contract_from_grain_contracts,
    build_training_example_from_schema_contract,
    dedupe_fn_by_dataset_from_grain_contracts,
    merge_grain_contracts_into_school_config,
    merge_grain_learner_id_alias_into_school_config,
    process_school_dataset,
    save_enriched_schema_contract,
    save_enriched_schema_contracts,
)
from edvise.genai.mapping.schema_contract import (
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
    "UNIQUE_VALUES_MAX_CARDINALITY",
    "apply_grain_dedup",
    "apply_grain_execution",
    "apply_term_order_from_contract",
    "apply_term_order_from_config",
    "term_order_column_for_clean_dataset",
    "term_order_fn_from_term_order_config",
    "build_dedupe_fn_from_grain_contract",
    "build_enriched_schema_contract_for_institution",
    "build_enriched_schema_contract_for_dataset",
    "build_schema_contract_from_grain_contracts",
    "dedupe_fn_by_dataset_from_grain_contracts",
    "build_training_example_from_schema_contract",
    "canonicalize_grain_contract_learner_id_alias",
    "merge_grain_contracts_into_school_config",
    "merge_grain_learner_id_alias_into_school_config",
    "parse_enriched_schema_contract_for_sma",
    "process_school_dataset",
    "save_enriched_schema_contract",
    "save_enriched_schema_contracts",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from edvise.genai.mapping.identity_agent.term_normalization.term_order import (
            apply_term_order_from_config as fn,
        )

        return fn
    if name in (
        "term_order_column_for_clean_dataset",
        "term_order_fn_from_term_order_config",
    ):
        from edvise.genai.mapping.identity_agent.term_normalization import (
            term_order as _term_order,
        )

        return getattr(_term_order, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
