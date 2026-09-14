"""Backward-compatible exports for instructional_modality canonicalization."""

from edvise.data_audit.es_categorical_map import (
    DELIVERY_METHOD_CATEGORIES,
    INSTRUCTIONAL_MODALITY_TO_PDP,
    es_modality_dummy_aliases,
    instructional_modality_series_to_pdp,
    instructional_modality_to_pdp_val,
)

__all__ = [
    "DELIVERY_METHOD_CATEGORIES",
    "INSTRUCTIONAL_MODALITY_TO_PDP",
    "es_modality_dummy_aliases",
    "instructional_modality_series_to_pdp",
    "instructional_modality_to_pdp_val",
]
