"""Map Edvise (ES) feature column names to PDP features-table keys."""

from __future__ import annotations

from dataclasses import fields

from edvise.configs.schema_type import is_edvise_schema
from edvise.feature_generation.column_names import (
    CohortInputColumns,
    CourseInputColumns,
    ES_COHORT_INPUT_COLUMNS,
    ES_COURSE_INPUT_COLUMNS,
    PDP_COHORT_INPUT_COLUMNS,
    PDP_COURSE_INPUT_COLUMNS,
)


# Whole-column aliases when snake_case mangles Edvise schema names (year1 -> year_1).
# Substring token replacement does not connect these (``pell_recipient_year1`` is not
# contained in ``pell_recipient_year_1``).
_ES_EXACT_COLUMN_ALIASES: dict[str, str] = {
    "pell_recipient_year_1": "student_is_pell_recipient_first_year",
    "pell_recipient_year1": "student_is_pell_recipient_first_year",
}

# Edvise-only columns that pass through to the modeling dataset (see ESCleanup).
ES_ONLY_FEATURES_TABLE_COLUMNS: tuple[str, ...] = (
    "intended_program_type",
    "declared_major_at_entry",
    "credits_earned_ap",
    "credits_earned_dual_enrollment",
    "term_degree",
    "term_degree_changed_prev_term",
)


def _add_es_to_pdp_tokens(
    mapping: dict[str, str],
    es_cols: CohortInputColumns | CourseInputColumns,
    pdp_cols: CohortInputColumns | CourseInputColumns,
) -> None:
    for f in fields(es_cols):
        es_val = getattr(es_cols, f.name)
        pdp_val = getattr(pdp_cols, f.name)
        if (
            isinstance(es_val, str)
            and isinstance(pdp_val, str)
            and es_val.lower() != pdp_val.lower()
        ):
            mapping[es_val.lower()] = pdp_val.lower()


def build_es_to_pdp_feature_token_map() -> dict[str, str]:
    """
    Build Edvise physical column token -> PDP token replacements for features-table lookup.

    Derived from :data:`ES_COHORT_INPUT_COLUMNS` / :data:`ES_COURSE_INPUT_COLUMNS` vs
    their PDP counterparts wherever both sides define the same logical field.
    """
    mapping: dict[str, str] = {}
    _add_es_to_pdp_tokens(mapping, ES_COHORT_INPUT_COLUMNS, PDP_COHORT_INPUT_COLUMNS)
    _add_es_to_pdp_tokens(mapping, ES_COURSE_INPUT_COLUMNS, PDP_COURSE_INPUT_COLUMNS)
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

    if col in _ES_EXACT_COLUMN_ALIASES:
        return _ES_EXACT_COLUMN_ALIASES[col]

    token_map = build_es_to_pdp_feature_token_map()
    for es_token in sorted(token_map, key=len, reverse=True):
        if es_token in col:
            col = col.replace(es_token, token_map[es_token])
    return col
