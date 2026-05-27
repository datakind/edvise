"""
Pandera validation diagnostics for SMA / pipeline Step 2d.

Writes a compact ``pandera_validation_errors.json`` under the run root (not ``run_log.json``).
Full ``lazy=True`` validation still runs once per entity; only a small sample of failure rows
is persisted so large datasets stay fast to report on.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PANDERA_VALIDATION_ERRORS_BASENAME = "pandera_validation_errors.json"
DEFAULT_FAILURE_CASES_SAMPLE_SIZE = 10


def _json_default(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def normalize_failure_cases(failure_cases: pd.DataFrame) -> pd.DataFrame:
    """Align Pandera failure_cases rows for grouping (same as sma_dev notebook)."""
    df = failure_cases.copy()
    required_cols = {"schema_context", "column", "check", "failure_case"}
    if not required_cols.issubset(df.columns):
        return df
    missing_col_mask = df["schema_context"].astype(str).eq("DataFrameSchema") & df[
        "check"
    ].astype(str).eq("column_in_dataframe")
    if missing_col_mask.any():
        df.loc[missing_col_mask, "column"] = df.loc[missing_col_mask, "failure_case"]
        df.loc[missing_col_mask, "check"] = "missing_column"
    return df


def summarize_failure_cases(failure_cases: pd.DataFrame) -> list[dict[str, Any]]:
    """Group failures by schema_context / column / check (compact, not per-row)."""
    if failure_cases.empty:
        return []
    normalized = normalize_failure_cases(failure_cases)
    group_cols = [
        c for c in ("schema_context", "column", "check") if c in normalized.columns
    ]
    if not group_cols:
        return []
    grouped = (
        normalized.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_failures")
        .sort_values("n_failures", ascending=False)
    )
    return [
        {k: _json_default(v) for k, v in row.items()}
        for row in grouped.to_dict(orient="records")
    ]


def sample_failure_cases(
    failure_cases: pd.DataFrame,
    *,
    sample_size: int = DEFAULT_FAILURE_CASES_SAMPLE_SIZE,
) -> list[dict[str, Any]]:
    if failure_cases.empty or sample_size <= 0:
        return []
    normalized = normalize_failure_cases(failure_cases)
    sample = normalized.head(sample_size)
    return [
        {k: _json_default(v) for k, v in row.items()}
        for row in sample.to_dict(orient="records")
    ]


def validate_entity_pandera(
    df: pd.DataFrame,
    schema: Any,
    label: str,
    *,
    failure_sample_size: int = DEFAULT_FAILURE_CASES_SAMPLE_SIZE,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run Pandera lazy validation; return one entity block for the JSON report."""
    import pandera

    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    out: dict[str, Any] = {
        "label": label,
        "status": "passed",
        "duration_s": 0.0,
        "row_count": int(len(df)),
        "failure_case_count": 0,
        "failure_summary": [],
        "failure_cases_sample": [],
        "error": None,
    }
    try:
        schema.validate(df, lazy=True)
        out["duration_s"] = round(time.perf_counter() - start, 3)
        log.info("Pandera [%s]: PASSED (%.2fs)", label, out["duration_s"])
        return out
    except pandera.errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        out["status"] = "failed"
        out["duration_s"] = round(time.perf_counter() - start, 3)
        out["failure_case_count"] = int(len(failure_cases))
        out["failure_summary"] = summarize_failure_cases(failure_cases)
        out["failure_cases_sample"] = sample_failure_cases(
            failure_cases, sample_size=failure_sample_size
        )
        summary_lines = ""
        if out["failure_summary"]:
            top = out["failure_summary"][:10]
            summary_lines = "\n" + pd.DataFrame(top).to_string(index=False)
        log.warning(
            "Pandera [%s]: FAILED — %d case(s) (%.2fs)%s",
            label,
            out["failure_case_count"],
            out["duration_s"],
            summary_lines,
        )
        return out
    except Exception as exc:
        out["status"] = "error"
        out["duration_s"] = round(time.perf_counter() - start, 3)
        out["error"] = str(exc)
        log.exception("Pandera [%s]: validation error", label)
        return out


def course_row_uniqueness_report(
    df: pd.DataFrame, *, logger: logging.Logger | None = None
) -> dict[str, Any]:
    from edvise.data_audit.schemas.raw_edvise_course import (
        course_output_row_uniqueness_violation_message,
        course_output_uniqueness_key_columns,
    )

    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    keys = course_output_uniqueness_key_columns(df)
    umsg = course_output_row_uniqueness_violation_message(df)
    elapsed = round(time.perf_counter() - start, 3)
    if umsg is not None:
        log.warning(
            "Course row uniqueness [%s]: FAILED (%.2fs)\n%s",
            keys,
            elapsed,
            umsg,
        )
        return {
            "status": "failed",
            "duration_s": elapsed,
            "keys": keys,
            "message": umsg,
        }
    log.info("Course row uniqueness [%s]: OK (%.2fs)", keys, elapsed)
    return {
        "status": "ok",
        "duration_s": elapsed,
        "keys": keys,
        "message": None,
    }


def build_pandera_validation_report(
    cohort_df: pd.DataFrame,
    course_df: pd.DataFrame,
    *,
    failure_sample_size: int = DEFAULT_FAILURE_CASES_SAMPLE_SIZE,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema

    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "failure_cases_sample_size": failure_sample_size,
        "cohort": validate_entity_pandera(
            cohort_df,
            RawEdviseStudentDataSchema,
            "cohort",
            failure_sample_size=failure_sample_size,
            logger=logger,
        ),
        "course": validate_entity_pandera(
            course_df,
            RawEdviseCourseDataSchema,
            "course",
            failure_sample_size=failure_sample_size,
            logger=logger,
        ),
        "course_row_uniqueness": course_row_uniqueness_report(course_df, logger=logger),
    }


def write_pandera_validation_errors(
    report_path: str | Path,
    cohort_df: pd.DataFrame,
    course_df: pd.DataFrame,
    *,
    failure_sample_size: int = DEFAULT_FAILURE_CASES_SAMPLE_SIZE,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run Step 2d validation and write ``pandera_validation_errors.json``."""
    log = logger or logging.getLogger(__name__)
    path = Path(report_path)
    report = build_pandera_validation_report(
        cohort_df,
        course_df,
        failure_sample_size=failure_sample_size,
        logger=logger,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    log.info("Wrote Pandera validation report -> %s", path)
    return report
