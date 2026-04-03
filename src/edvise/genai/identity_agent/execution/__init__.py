"""
IdentityAgent **execution** — grain transforms + frozen schema contract (SMA).

- :mod:`grain_transforms` — in-memory dedup / term order; ``dedupe_fn`` factory for ``clean_dataset``.
- :mod:`schema_contract_executor` — merge grain into school config; call SMA preprocessing.

Preprocessing already runs ``clean_dataset`` (dedupe_fn, unique-key dedupe, term_order_fn) when
those kwargs are set. For grain ``no_dedup``, duplicate keys may still fail the final
uniqueness check unless you adjust enforcement upstream.
"""

from .grain_transforms import (
    apply_grain_dedup,
    apply_grain_execution,
    apply_grain_term_order,
    build_dedupe_fn_from_grain_contract,
)
from .schema_contract_executor import (
    build_schema_contract_from_grain_contracts,
    merge_grain_contracts_into_school_config,
)

__all__ = [
    "apply_grain_dedup",
    "apply_grain_execution",
    "apply_grain_term_order",
    "build_dedupe_fn_from_grain_contract",
    "build_schema_contract_from_grain_contracts",
    "merge_grain_contracts_into_school_config",
]
