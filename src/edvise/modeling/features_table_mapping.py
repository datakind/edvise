"""Map Edvise (ES) feature column names to PDP features-table keys."""

from __future__ import annotations

from dataclasses import fields

from edvise.configs.schema_type import is_edvise_schema
from edvise.feature_generation.column_names import (
    ES_COHORT_INPUT_COLUMNS,
    ES_COURSE_INPUT_COLUMNS,
    PDP_COHORT_INPUT_COLUMNS,
    PDP_COURSE_INPUT_COLUMNS,
)


# Edvise student renames not represented on :class:`CohortInputColumns`.
_ES_ADDITIONAL_STUDENT_TOKEN_MAP: dict[str, str] = {
    "learner_age": "student_age",
    "first_generation_status": "first_gen",
}

# Edvise-only cohort columns that pass through to the modeling dataset (see ESCleanup).
ES_ONLY_FEATURES_TABLE_COLUMNS: tuple[str, ...] = (
    "intended_program_type",
    "declared_major_at_entry",
    "credits_earned_ap",
    "credits_earned_dual_enrollment",
)

# PDP table keys for Edvise columns renamed via :data:`_ES_ADDITIONAL_STUDENT_TOKEN_MAP`.
ES_MAPPED_FEATURES_TABLE_COLUMNS: tuple[str, ...] = (
    "student_age",
    "first_gen",
)


def build_es_to_pdp_feature_token_map() -> dict[str, str]:
    """
    Build Edvise physical column token -> PDP token replacements for features-table lookup.

    Derived from :data:`ES_COHORT_INPUT_COLUMNS` / :data:`ES_COURSE_INPUT_COLUMNS` vs
    their PDP counterparts wherever both sides define the same logical field, plus
    :data:`_ES_ADDITIONAL_STUDENT_TOKEN_MAP`.
    """
    mapping: dict[str, str] = {}
    pairs = [
        (ES_COHORT_INPUT_COLUMNS, PDP_COHORT_INPUT_COLUMNS),
        (ES_COURSE_INPUT_COLUMNS, PDP_COURSE_INPUT_COLUMNS),
    ]
    for es_cols, pdp_cols in pairs:
        for f in fields(es_cols):
            es_val = getattr(es_cols, f.name)
            pdp_val = getattr(pdp_cols, f.name)
            if (
                isinstance(es_val, str)
                and isinstance(pdp_val, str)
                and es_val.lower() != pdp_val.lower()
            ):
                mapping[es_val.lower()] = pdp_val.lower()
    mapping.update(_ES_ADDITIONAL_STUDENT_TOKEN_MAP)
    return mapping


def map_feature_col_for_features_table(
    feature_col: str,
    schema_type: str | None = None,
) -> str:
    """
    Normalize a modeling feature column name for lookup in the shared features table.

    For Edvise schema types, replace embedded Edvise physical column tokens with their
    PDP equivalents (e.g. ``instructional_modality`` -> ``delivery_method`` in
    ``num_courses_instructional_modality_f``).
    """
    col = feature_col.lower()
    if not schema_type or not is_edvise_schema(schema_type):
        return col

    token_map = build_es_to_pdp_feature_token_map()
    for es_token in sorted(token_map, key=len, reverse=True):
        if es_token in col:
            col = col.replace(es_token, token_map[es_token])
    return col
