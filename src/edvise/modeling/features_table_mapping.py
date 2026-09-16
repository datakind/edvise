"""Map Edvise (ES) feature column names to PDP features-table keys."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

from edvise.shared.schema_type import is_edvise_schema
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
    # Whole-column: a trailing "_no" dummy alias would also rewrite gen-ed Y/N.
    "num_courses_gateway_or_developmental_flag_no": (
        "num_courses_math_or_english_gateway_na"
    ),
    "frac_courses_gateway_or_developmental_flag_no": (
        "frac_courses_math_or_english_gateway_na"
    ),
    "cumfrac_num_courses_gateway_or_developmental_flag_no": (
        "cumfrac_num_courses_math_or_english_gateway_na"
    ),
}

# Dummy-value suffixes after get_dummies, mapped onto the PDP features-table set.
# Applied only as a trailing ``_{token}`` so single-letter codes (s/u) cannot
# rewrite earlier parts of the column name.
_ES_DUMMY_VALUE_ALIASES: dict[str, str] = {
    "gateway_english": "e",
    "gateway_math": "m",
    "pass": "p",
    "sat": "p",
    "s": "p",
    "unsat": "f",
    "u": "f",
    "wd": "w",
    "ip": "i",
    "nr": "m",
    "ng": "m",
    "hybrid": "h",
    "online": "o",
    "in_person": "f",
    "face_to_face": "f",
    "f2f": "f",
    # Instructional_modality suffixes. Add more as other ES schools send values.
    "face_to_face_and_online": "h",
    "online_internet_or_web": "o",
    "web_based": "o",
    "fully_online": "o",
    "hybrid_asynchronous": "h",
    "hybrid_synchronous": "h",
    "online_asynchronous": "o",
    "online_mix": "o",
    "online_synchronous": "o",
    "online_that_is_hybrid": "h",
    "web_enhanced": "f",
    # Instructor appointment suffixes. Add more as other ES schools send values.
    "adjunct_assistant_professor": "pt",
    "adjunct_instructor": "pt",
    "adjunct_professor": "pt",
    "teaching_assistant": "pt",
    "research_assistant_professor": "ft",
    "visiting_assistant_professor": "ft",
    "visiting_research_asst_prof": "ft",
    "visiting_professor": "ft",
    "assistant_professor": "ft",
    "associate_professor": "ft",
    "instructor": "ft",
    "professor": "ft",
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


@lru_cache(maxsize=1)
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


def _apply_dummy_value_alias(col: str) -> str:
    """Replace a trailing ES dummy suffix with its PDP features-table counterpart."""
    for es_val in sorted(_ES_DUMMY_VALUE_ALIASES, key=len, reverse=True):
        suffix = f"_{es_val}"
        if col.endswith(suffix):
            return col[: -len(suffix)] + f"_{_ES_DUMMY_VALUE_ALIASES[es_val]}"
    return col


def map_feature_col_for_features_table(
    feature_col: str,
    schema_type: str | None = None,
) -> str:
    """
    Normalize a modeling feature column name for lookup in the shared features table.

    For Edvise schema types, replace embedded Edvise physical column tokens with their
    PDP equivalents (e.g. ``instructional_modality`` -> ``delivery_method`` in
    ``num_courses_instructional_modality_f``), then dummy-value suffixes
    (e.g. ``gateway_english`` -> ``e``, ``course_grade_s`` -> ``course_grade_p``).
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
    return _apply_dummy_value_alias(col)
