import faker
import pandas as pd
import pytest

from edvise.data_audit.schemas import RawEdviseStudentDataSchema
from edvise.synth_generation.es import raw_student

FAKER = faker.Faker()
FAKER.add_provider(raw_student.Provider)


@pytest.mark.parametrize("include_optionals", [True, False])
def test_raw_student_record(include_optionals):
    obs = FAKER.raw_student_record(include_optionals=include_optionals)
    assert obs and isinstance(obs, dict)
    assert {"learner_id", "entry_year", "entry_term"} <= set(obs)
    df_obs = pd.DataFrame([obs])
    obs_valid = RawEdviseStudentDataSchema.validate(df_obs, lazy=True)
    assert isinstance(obs_valid, pd.DataFrame)


def test_multiple_raw_student_records():
    student_records = [
        FAKER.raw_student_record(include_optionals=True) for _ in range(10)
    ]
    # Ensure unique learner_ids for schema unique constraint
    for i, rec in enumerate(student_records):
        rec["learner_id"] = f"{10000 + i}"
    df_students = pd.DataFrame(student_records)
    obs_valid = RawEdviseStudentDataSchema.validate(df_students, lazy=True)
    assert isinstance(obs_valid, pd.DataFrame)
