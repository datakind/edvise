"""Academic-year calendar shared by feature generation, targets, and model prep.

Raw term columns use a fixed vocabulary (FALL, WINTER, SPRING, SUMMER). Which of
those terms opens the academic year is an institution setting, not a property of
that vocabulary.
"""

from __future__ import annotations

from typing import Literal

DEFAULT_FIRST_TERM_OF_YEAR: Literal["FALL", "SUMMER"] = "FALL"

# Within one YYYY-YY label, in chronological order.
_TERM_ORDER_BY_FIRST_TERM: dict[str, tuple[str, ...]] = {
    "FALL": ("FALL", "WINTER", "SPRING", "SUMMER"),
    "SUMMER": ("SUMMER", "FALL", "WINTER", "SPRING"),
}

CANONICAL_ACADEMIC_TERMS = frozenset({"FALL", "WINTER", "SPRING", "SUMMER"})


def academic_term_order(first_term_of_year: str) -> tuple[str, ...]:
    """Season order within one academic year, earliest term first."""
    key = str(first_term_of_year).upper()
    try:
        return _TERM_ORDER_BY_FIRST_TERM[key]
    except KeyError as exc:
        raise ValueError(
            f"first_term_of_year must be 'FALL' or 'SUMMER', got {first_term_of_year!r}"
        ) from exc


def summer_is_end_of_academic_year(first_term_of_year: str) -> bool:
    """True when Summer is the last term of YYYY-YY (the year starts in Fall)."""
    return str(first_term_of_year).upper() != "SUMMER"


def cohort_term_sort_ranks(first_term_of_year: str) -> dict[str, int]:
    """1-based lowercase season ranks for sorting cohort labels such as ``fall 2024-25``."""
    return {
        term.lower(): i + 1
        for i, term in enumerate(academic_term_order(first_term_of_year))
    }
