"""Tests for pandera_validation_errors.json report helper."""

import json
from pathlib import Path

import pandas as pd
from pandera import Check, Column, DataFrameSchema

from edvise.data_audit.schemas.pandera_validation_report import (
    sample_failure_cases,
    summarize_failure_cases,
)


def test_sample_failure_cases_caps_at_ten() -> None:
    failure_cases = pd.DataFrame(
        {
            "schema_context": ["Column"] * 25,
            "column": ["age"] * 25,
            "check": ["greater_than_or_equal_to"] * 25,
            "failure_case": list(range(25)),
        }
    )
    sample = sample_failure_cases(failure_cases, sample_size=10)
    assert len(sample) == 10
    assert sample[0]["failure_case"] == 0


def test_summarize_failure_cases_groups() -> None:
    failure_cases = pd.DataFrame(
        {
            "schema_context": ["Column", "Column", "Column"],
            "column": ["age", "age", "score"],
            "check": ["ge", "ge", "ge"],
            "failure_case": [1, 2, 3],
        }
    )
    summary = summarize_failure_cases(failure_cases)
    assert len(summary) == 2
    age_row = next(r for r in summary if r["column"] == "age")
    assert age_row["n_failures"] == 2


def test_validate_entity_pandera_writes_sample(tmp_path: Path) -> None:
    from edvise.data_audit.schemas.pandera_validation_report import (
        validate_entity_pandera,
    )

    schema = DataFrameSchema(
        {"x": Column(int, Check.ge(0), nullable=False)},
    )
    ok_df = pd.DataFrame({"x": [1, 2]})
    bad_df = pd.DataFrame({"x": list(range(-15, 0))})

    passed = validate_entity_pandera(ok_df, schema, "ok")
    failed = validate_entity_pandera(bad_df, schema, "bad", failure_sample_size=10)

    assert passed["status"] == "passed"
    assert failed["status"] == "failed"
    assert failed["failure_case_count"] >= 11
    assert len(failed["failure_cases_sample"]) == 10
    assert failed["failure_summary"]

    out_path = tmp_path / "pandera_validation_errors.json"
    out_path.write_text(json.dumps({"bad": failed}, indent=2), encoding="utf-8")
    loaded = json.loads(out_path.read_text())
    assert len(loaded["bad"]["failure_cases_sample"]) == 10
