from __future__ import annotations

from typing import Any

from .grain_transforms import (
    apply_grain_dedup,
    apply_grain_execution,
    apply_grain_term_order,
    build_dedupe_fn_from_grain_contract,
    canonicalize_grain_contract_student_id_alias,
)
from .schema_contract_executor import (
    UNIQUE_VALUES_MAX_CARDINALITY,
    build_schema_contract_from_grain_contracts,
    build_training_example_from_schema_contract,
    merge_grain_contracts_into_school_config,
    merge_grain_student_id_alias_into_school_config,
    process_all_schools,
    process_school_dataset,
    save_enriched_schema_contracts,
)

__all__ = [
    "UNIQUE_VALUES_MAX_CARDINALITY",
    "apply_grain_dedup",
    "apply_grain_execution",
    "apply_grain_term_order",
    "apply_term_order_from_config",
    "term_order_column_for_clean_dataset",
    "term_order_fn_from_term_order_config",
    "build_dedupe_fn_from_grain_contract",
    "build_schema_contract_from_grain_contracts",
    "build_training_example_from_schema_contract",
    "canonicalize_grain_contract_student_id_alias",
    "merge_grain_contracts_into_school_config",
    "merge_grain_student_id_alias_into_school_config",
    "process_all_schools",
    "process_school_dataset",
    "save_enriched_schema_contracts",
]


def __getattr__(name: str) -> Any:
    if name == "apply_term_order_from_config":
        from edvise.genai.identity_agent.term_normalization.utilities import (
            apply_term_order_from_config as fn,
        )

        return fn
    if name in (
        "term_order_column_for_clean_dataset",
        "term_order_fn_from_term_order_config",
    ):
        from edvise.genai.identity_agent.term_normalization import utilities as u

        return getattr(u, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
