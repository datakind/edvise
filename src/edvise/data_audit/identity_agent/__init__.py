"""Identity-oriented data utilities (deduplication, future record linking)."""

from .deduplication import (
    apply_key_collision_dedupe_from_spec,
    drop_duplicate_keys,
    drop_exact_row_duplicates,
    resolve_key_collisions,
    suffix_disambiguate_within_keys,
)

__all__ = [
    "apply_key_collision_dedupe_from_spec",
    "drop_duplicate_keys",
    "drop_exact_row_duplicates",
    "resolve_key_collisions",
    "suffix_disambiguate_within_keys",
]
