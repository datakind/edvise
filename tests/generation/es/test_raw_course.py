import faker
import pandas as pd
import pytest

from edvise.data_audit.schemas import RawEdviseCourseDataSchema
from edvise.synth_generation.es import raw_course, raw_student

FAKER = faker.Faker()
FAKER.add_provider(raw_student.Provider)
FAKER.add_provider(raw_course.Provider)


@pytest.mark.parametrize("include_optionals", [True, False])
def test_raw_course_record(include_optionals):
    obs = FAKER.raw_course_record(include_optionals=include_optionals)
    assert obs and isinstance(obs, dict)
    assert {
        "learner_id",
        "academic_year",
        "academic_term",
        "course_prefix",
        "course_number",
        "grade",
        "course_credits_attempted",
        "course_credits_earned",
    } <= set(obs)
    df_obs = pd.DataFrame([obs])
    obs_valid = RawEdviseCourseDataSchema.validate(df_obs, lazy=True)
    assert isinstance(obs_valid, pd.DataFrame)


def test_raw_course_record_from_student():
    student = FAKER.raw_student_record(include_optionals=True)
    course = FAKER.raw_course_record(student_record=student, include_optionals=True)
    assert course["learner_id"] == student["learner_id"]
    df_obs = pd.DataFrame([course])
    obs_valid = RawEdviseCourseDataSchema.validate(df_obs, lazy=True)
    assert isinstance(obs_valid, pd.DataFrame)
