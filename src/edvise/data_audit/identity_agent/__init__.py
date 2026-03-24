"""Identity-oriented data utilities (deduplication, future record linking)."""

from .deduplication import (
    drop_duplicate_keys,
    drop_exact_row_duplicates,
    resolve_key_collisions,
    suffix_disambiguate_within_keys,
)

__all__ = [
    "drop_duplicate_keys",
    "drop_exact_row_duplicates",
    "resolve_key_collisions",
    "suffix_disambiguate_within_keys",
]
