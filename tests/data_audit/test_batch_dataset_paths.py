"""Tests for ES batch cohort/course path resolution in data_audit."""

from __future__ import annotations

from pathlib import Path

from edvise.data_audit import batch_dataset_paths as m


def test_resolve_es_raw_dataset_paths_substring_match(tmp_path: Path) -> None:
    student = tmp_path / "1782424164337_2025-09-19_CCC Student File.csv"
    course = tmp_path / "1782424164335_2025-09-19_CCC Course File.csv"
    student.write_text("a\n", encoding="utf-8")
    course.write_text("b\n", encoding="utf-8")

    cohort_path, course_path = m.resolve_es_raw_dataset_paths(
        str(tmp_path),
        raw_cohort_name="2025-09-19_CCC Student File.csv",
        raw_course_name="2025-09-19_CCC Course File.csv",
    )
    assert cohort_path == str(student)
    assert course_path == str(course)
