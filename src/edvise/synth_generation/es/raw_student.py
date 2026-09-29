"""Faker provider for raw Edvise (ES) student / cohort records."""

from __future__ import annotations

import typing as t

from edvise.synth_generation.shared import SynthFieldMixin


class Provider(SynthFieldMixin):
    def raw_student_record(
        self,
        *,
        min_entry_yr: int = 2010,
        max_entry_yr: t.Optional[int] = None,
        learner_id: t.Optional[str] = None,
        include_optionals: bool = True,
    ) -> dict[str, object]:
        """
        Build one raw ES student record.

        Required keys always present. When ``include_optionals`` is True, optional
        schema columns are filled so downstream feature generation exercises the
        fuller ES surface.
        """
        record: dict[str, object] = {
            "learner_id": learner_id
            if learner_id is not None
            else self.synth_student_id(),
            "entry_year": self.synth_academic_year(
                min_yr=min_entry_yr, max_yr=max_entry_yr
            ),
            "entry_term": self.synth_term(),
            "enrollment_type": self.enrollment_type(),
            "intended_program_type": self.intended_program_type(),
            "declared_major_at_entry": self.synth_cip(),
        }
        if include_optionals:
            record.update(self._optional_student_fields())
        return record

    def _optional_student_fields(self) -> dict[str, object]:
        has_conferral = self.generator.random.random() < 0.3
        return {
            "matriculation_date": self.synth_course_date(),
            "learner_age": self.synth_student_age(),
            "race": self.synth_race(),
            "ethnicity": self.synth_ethnicity(),
            "gender": self.synth_gender(),
            "first_generation_status": self.first_generation_status(),
            "pell_recipient_year1": self.synth_pell_yn(),
            "incarcerated_status": self.incarcerated_status(),
            "military_status": self.military_status(),
            "employment_status": self.employment_status(),
            "disability_status": self.disability_status(),
            "bachelors_degree_conferral_date": (
                self.synth_course_date() if has_conferral else None
            ),
            "associates_degree_conferral_date": (
                self.synth_course_date() if has_conferral else None
            ),
            "conferred_credential_type": (
                self.intended_program_type() if has_conferral else None
            ),
            "major_at_completion": self.synth_cip() if has_conferral else None,
            "certificate1_date": None,
            "certificate2_date": None,
            "certificate3_date": None,
            "credits_earned_ap": self.synth_credits(min_value=0.0, max_value=30.0),
            "credits_earned_dual_enrollment": self.synth_credits(
                min_value=0.0, max_value=30.0
            ),
        }

    def enrollment_type(self) -> str:
        return self.random_element(["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"])

    def intended_program_type(self) -> str:
        return self.random_element(
            [
                "Associate Degree",
                "Bachelor's Degree",
                "Undergraduate Certificate or Diploma Program",
                "Non- Credential Program (Preparatory Coursework/Teach Certification)",
            ]
        )

    def first_generation_status(self) -> str:
        return self.random_element(["P", "C", "A", "B", "Y", "N"])

    def incarcerated_status(self) -> str:
        return self.random_element(["Y", "P", "N"])

    def military_status(self) -> str:
        return self.random_element(["-1", "0", "1", "2"])

    def employment_status(self) -> str:
        return self.random_element(["-1", "0", "1", "2", "3", "4"])

    def disability_status(self) -> str:
        return self.random_element(["Y", "N"])
