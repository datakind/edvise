"""Tests for :mod:`edvise.dataio.genai_eda_inputs`."""

from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest

from edvise.dataio import genai_eda_inputs as gei
from edvise.genai.mapping.state._sql import lit, qualified_table


def test_lit_escape_used_for_sql() -> None:
    assert lit("a'b") == "'a''b'"


def test_pipeline_input_parquet_paths_layout() -> None:
    cohort, course = gei.pipeline_input_parquet_paths(
        catalog="dev_sst_02",
        institution_id="hostos_cc",
        execute_run_id="ex-1",
    )
    assert cohort == (
        "/Volumes/dev_sst_02/hostos_cc_silver/silver_volume/"
        "genai_mapping/runs/execute/ex-1/pipeline_input/cohort.parquet"
    )
    assert course.endswith("/pipeline_input/course.parquet")


def test_lookup_rejects_unsafe_batch_id() -> None:
    with pytest.raises(ValueError, match="Invalid batch_id"):
        gei.lookup_genai_execute_runs_for_batch(
            mock.Mock(),
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_id="test_school",
            batch_id="not-a-batch",
        )


def test_lookup_builds_pipeline_runs_sql() -> None:
    workspace = mock.Mock()
    with mock.patch.object(gei, "_execute_sql", return_value=[]) as exec_sql:
        gei.lookup_genai_execute_runs_for_batch(
            workspace,
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_id="edison_state_cc",
            batch_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

    statement = exec_sql.call_args.kwargs["statement"]
    assert qualified_table("dev_sst_02", "pipeline_runs") in statement
    assert "gcs_uploads/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in statement
    assert "execute_run_id IS NOT NULL" in statement
    assert "ORDER BY updated_at DESC" in statement


def test_resolve_returns_none_when_no_pipeline_rows() -> None:
    workspace = mock.Mock()
    with mock.patch.object(gei, "lookup_genai_execute_runs_for_batch", return_value=[]):
        result = gei.resolve_genai_mapped_inputs_for_batch(
            workspace,
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_name="Test School",
            batch_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    assert result is None


def test_resolve_returns_none_when_only_running_rows() -> None:
    workspace = mock.Mock()
    rows = [
        {
            "execute_run_id": "ex-1",
            "status": "running",
            "updated_at": "2026-01-01",
            "input_file_paths": "gcs_uploads/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        }
    ]
    with mock.patch.object(
        gei, "lookup_genai_execute_runs_for_batch", return_value=rows
    ):
        result = gei.resolve_genai_mapped_inputs_for_batch(
            workspace,
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_name="Test School",
            batch_id="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
    assert result is None


def test_resolve_ready_loads_parquets() -> None:
    workspace = mock.Mock()
    rows = [
        {
            "execute_run_id": "ex-ready",
            "status": "complete",
            "updated_at": "2026-01-02",
            "input_file_paths": ".../gcs_uploads/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/...",
        }
    ]
    cohort = pd.DataFrame({"learner_id": ["L1"], "entry_year": ["2020-21"]})
    course = pd.DataFrame({"learner_id": ["L1"], "grade": ["3.0"]})

    with (
        mock.patch.object(gei, "lookup_genai_execute_runs_for_batch", return_value=rows),
        mock.patch.object(
            gei,
            "_read_volume_parquet",
            side_effect=[cohort, course],
        ),
    ):
        result = gei.resolve_genai_mapped_inputs_for_batch(
            workspace,
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_name="Test School",
            batch_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )

    assert result is not None
    assert result.execute_run_id == "ex-ready"
    assert list(result.df_cohort["learner_id"]) == ["L1"]
    assert result.df_course is not None


def test_resolve_raises_when_parquet_load_fails() -> None:
    workspace = mock.Mock()
    rows = [
        {
            "execute_run_id": "ex-ready",
            "status": "complete",
            "updated_at": "2026-01-02",
            "input_file_paths": ".../gcs_uploads/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/...",
        }
    ]
    with (
        mock.patch.object(gei, "lookup_genai_execute_runs_for_batch", return_value=rows),
        mock.patch.object(
            gei, "_read_volume_parquet", side_effect=FileNotFoundError("missing")
        ),
        pytest.raises(FileNotFoundError, match="missing"),
    ):
        gei.resolve_genai_mapped_inputs_for_batch(
            workspace,
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_name="Test School",
            batch_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )


def test_resolve_returns_none_when_cohort_empty() -> None:
    workspace = mock.Mock()
    rows = [
        {
            "execute_run_id": "ex-ready",
            "status": "complete",
            "updated_at": "2026-01-02",
            "input_file_paths": ".../gcs_uploads/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/...",
        }
    ]
    with (
        mock.patch.object(gei, "lookup_genai_execute_runs_for_batch", return_value=rows),
        mock.patch.object(
            gei,
            "_read_volume_parquet",
            side_effect=[pd.DataFrame(), pd.DataFrame({"learner_id": ["L1"]})],
        ),
    ):
        result = gei.resolve_genai_mapped_inputs_for_batch(
            workspace,
            warehouse_id="wh",
            catalog="dev_sst_02",
            institution_name="Test School",
            batch_id="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
    assert result is None
