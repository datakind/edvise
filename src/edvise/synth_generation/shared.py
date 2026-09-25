"""Shared Faker field helpers for PDP and Edvise synth providers."""

from __future__ import annotations

import typing as t
from datetime import date

from faker.providers import BaseProvider

from edvise.data_audit.schemas._edvise_shared import (
    LEARNER_AGE_BUCKETS,
    TERM_CATEGORIES,
)
from edvise.utils.data_cleaning import convert_to_snake_case

RACE_CATEGORIES = (
    "NONRESIDENT ALIEN",
    "AMERICAN INDIAN OR ALASKA NATIVE",
    "ASIAN",
    "BLACK OR AFRICAN AMERICAN",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
    "WHITE",
    "HISPANIC",
    "TWO OR MORE RACES",
    "UNKNOWN",
)

ETHNICITY_CATEGORIES = ("H", "N", "UK")
GENDER_CATEGORIES = ("M", "F", "P", "X", "UK")
PELL_YN_CATEGORIES = ("Y", "N")


class SynthFieldMixin(BaseProvider):
    """Common generators used by PDP and Edvise raw-record providers."""

    def synth_student_id(self) -> str:
        return self.numerify("#####!")

    def synth_institution_id(self) -> str:
        return self.numerify("#####!")

    def synth_academic_year(
        self, min_yr: int = 2010, max_yr: t.Optional[int] = None
    ) -> str:
        start_dt = self.generator.date_between(
            start_date=date(min_yr, 1, 1),
            end_date=(date(max_yr, 1, 1) if max_yr is not None else "today"),
        )
        end_yr = f"{start_dt.year + 1}"[2:]
        return f"{start_dt.year}-{end_yr}"

    def synth_term(self) -> str:
        return self.random_element(list(TERM_CATEGORIES))

    def synth_student_age(self) -> str:
        return self.random_element(list(LEARNER_AGE_BUCKETS))

    def synth_race(self) -> str:
        return self.random_element(list(RACE_CATEGORIES))

    def synth_ethnicity(self) -> str:
        return self.random_element(list(ETHNICITY_CATEGORIES))

    def synth_gender(self) -> str:
        return self.random_element(list(GENDER_CATEGORIES))

    def synth_pell_yn(self) -> str:
        return self.random_element(list(PELL_YN_CATEGORIES))

    def synth_cip(self) -> str:
        return self.numerify("##.####")

    def synth_course_prefix(self) -> str:
        return self.lexify("????").upper()

    def synth_course_number(self) -> str:
        return self.numerify("##!")

    def synth_section_id(self) -> str:
        return self.numerify("##!.#")

    def synth_course_title(self) -> str:
        return " ".join(self.generator.words(nb=3, part_of_speech="noun")).title()

    def synth_credits(self, min_value: float = 0.0, max_value: float = 20.0) -> float:
        return self.generator.pyfloat(  # type: ignore[no-any-return]
            min_value=min_value,
            max_value=max(max_value, min_value + 1e-3),
            right_digits=1,
        )

    def synth_course_date(
        self, min_yr: int = 2010, max_yr: t.Optional[int] = None
    ) -> date:
        end = date(max_yr, 1, 1) if max_yr is not None else "today"
        return self.generator.date_between(  # type: ignore[no-any-return]
            start_date=date(min_yr, 1, 1), end_date=end
        )


def maybe_snake_case_keys(
    record: dict[str, object], *, normalize_col_names: bool
) -> dict[str, object]:
    """Optionally convert title-case PDP keys to snake_case."""
    if not normalize_col_names:
        return record
    return {convert_to_snake_case(key): val for key, val in record.items()}
