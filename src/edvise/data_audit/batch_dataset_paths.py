"""Resolve ES cohort/course CSV paths from batch bronze landing dirs."""

from __future__ import annotations

from edvise.dataio.batch_dataset_paths import resolve_dataset_file_in_batch_dir


def resolve_es_raw_dataset_paths(
    bronze_batch_dir: str,
    *,
    raw_cohort_name: str,
    raw_course_name: str,
) -> tuple[str, str]:
    """
    Resolve ES cohort/course CSV paths under ``gcs_uploads/{batch_id}/``.

    Called from ``es_data_audit`` for edvise_id inference batches after ingest
    has landed validated files in ``bronze_batch_dir``.
    """
    batch_dir = (bronze_batch_dir or "").strip()
    cohort_name = (raw_cohort_name or "").strip()
    course_name = (raw_course_name or "").strip()
    if not batch_dir:
        raise ValueError(
            "bronze_batch_dir is required to resolve ES batch cohort/course paths."
        )
    if not cohort_name:
        raise ValueError("raw_cohort must be set in ES config.")
    if not course_name:
        raise ValueError("raw_course must be set in ES config.")

    cohort_path = resolve_dataset_file_in_batch_dir(batch_dir, cohort_name)
    if cohort_path is None:
        raise FileNotFoundError(
            f"Cohort file {cohort_name!r} not found under {batch_dir!r}."
        )
    course_path = resolve_dataset_file_in_batch_dir(batch_dir, course_name)
    if course_path is None:
        raise FileNotFoundError(
            f"Course file {course_name!r} not found under {batch_dir!r}."
        )
    return cohort_path, course_path
