"""Faker provider for raw Edvise (ES) course records."""

from __future__ import annotations

import typing as t

from edvise.data_audit.schemas.raw_edvise_course import ALLOWED_LETTER_GRADES
from edvise.synth_generation.shared import SynthFieldMixin

_ES_GRADES = tuple(sorted(ALLOWED_LETTER_GRADES))


class Provider(SynthFieldMixin):
    def raw_course_record(
        self,
        student_record: t.Optional[dict] = None,
        *,
        include_optionals: bool = True,
    ) -> dict[str, object]:
        """
        Build one raw ES course record.

        When ``student_record`` is provided, ``learner_id`` is taken from it and
        ``academic_year`` is constrained to start at or after the student's
        ``entry_year``.
        """
        if student_record is not None:
            learner_id = student_record["learner_id"]
            entry_year = student_record.get("entry_year")
            min_yr = (
                int(str(entry_year).split("-")[0])
                if isinstance(entry_year, str) and "-" in entry_year
                else 2010
            )
        else:
            learner_id = self.synth_student_id()
            min_yr = 2010

        academic_year = self.synth_academic_year(min_yr=min_yr)
        academic_term = self.synth_term()
        course_prefix = self.synth_course_prefix()
        course_number = self.synth_course_number()
        credits_attempted = self.synth_credits(min_value=1.0, max_value=20.0)
        credits_earned = self.synth_credits(min_value=0.0, max_value=credits_attempted)

        record: dict[str, object] = {
            "learner_id": learner_id,
            "academic_year": academic_year,
            "academic_term": academic_term,
            "course_prefix": course_prefix,
            "course_number": course_number,
            "grade": self.grade(),
            "course_credits_attempted": credits_attempted,
            "course_credits_earned": credits_earned,
        }
        if include_optionals:
            record.update(
                self._optional_course_fields(
                    academic_year=academic_year,
                    academic_term=academic_term,
                    course_prefix=course_prefix,
                )
            )
        return record

    def _optional_course_fields(
        self,
        *,
        academic_year: str,
        academic_term: str,
        course_prefix: str,
    ) -> dict[str, object]:
        min_course_yr = int(academic_year.split("-")[0])
        max_course_yr = min_course_yr + 1
        begin = self.synth_course_date(min_yr=min_course_yr, max_yr=max_course_yr)
        end = self.synth_course_date(min_yr=min_course_yr, max_yr=max_course_yr)
        begin, end = sorted([begin, end])
        return {
            "source_term_key": f"{academic_year}|{academic_term}|1",
            "course_section_id": self.synth_section_id(),
            "course_title": self.synth_course_title(),
            "department": course_prefix,
            "instructional_format": self.random_element(
                ["LECTURE", "LAB", "SEMINAR", "ONLINE", "HYBRID"]
            ),
            "academic_level": self.random_element(
                ["UNDERGRADUATE", "GRADUATE", "DEVELOPMENTAL"]
            ),
            "course_begin_date": begin,
            "course_end_date": end,
            "instructional_modality": self.random_element(
                ["IN_PERSON", "ONLINE", "HYBRID"]
            ),
            "gen_ed_flag": self.random_element(["Y", "N"]),
            "prerequisite_flag": self.random_element(["Y", "N"]),
            "instructor_appointment_status": self.random_element(["PT", "FT"]),
            "gateway_or_developmental_flag": self.random_element(
                ["GATEWAY", "DEVELOPMENTAL", "NEITHER", "NA"]
            ),
            "course_section_size": float(self.random_int(min=5, max=120)),
            "term_degree": self.random_element(
                ["Associate Degree", "Bachelor's Degree", None]
            ),
            "term_declared_major": self.synth_cip(),
            "intent_to_transfer_flag": self.random_element(["Y", "N", None]),
            "term_pell_recipient": self.synth_pell_yn(),
        }

    def grade(self) -> str:
        return self.random_element(list(_ES_GRADES))
