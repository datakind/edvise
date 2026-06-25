"""Tests for inference model artifact resolution."""

from __future__ import annotations

from unittest import mock

from edvise.dataio import inference_model_artifacts as m


def test_silver_run_root_for_model() -> None:
    assert m.silver_run_root_for_model(
        db_workspace="edvise",
        databricks_institution_name="acme",
        model_run_id="run-123",
    ) == "/Volumes/edvise/acme_silver/silver_volume/run-123"


def test_resolve_es_inference_artifacts() -> None:
    with (
        mock.patch.object(
            m,
            "get_latest_uc_model_run_id",
            return_value="run-abc",
        ) as get_run_id,
        mock.patch.object(
            m,
            "find_file_in_run_folder",
            return_value="/Volumes/edvise/acme_silver/silver_volume/run-abc/config.toml",
        ) as find_file,
    ):
        artifacts = m.resolve_es_inference_artifacts(
            model_name="retention_model",
            db_workspace="edvise",
            databricks_institution_name="acme",
        )

    get_run_id.assert_called_once_with(
        model_name="retention_model",
        workspace="edvise",
        institution="acme",
    )
    find_file.assert_called_once_with(
        "/Volumes/edvise/acme_silver/silver_volume/run-abc",
        keyword="config",
    )
    assert artifacts.model_run_id == "run-abc"
    assert artifacts.config_file_path.endswith("config.toml")
    assert artifacts.silver_run_root == "/Volumes/edvise/acme_silver/silver_volume/run-abc"


def test_resolve_legacy_inference_artifacts() -> None:
    with (
        mock.patch.object(m, "get_latest_uc_model_run_id", return_value="run-legacy"),
        mock.patch.object(
            m,
            "find_file_in_run_folder",
            side_effect=[
                "/Volumes/edvise/acme_silver/silver_volume/run-legacy/config.toml",
                "/Volumes/edvise/acme_silver/silver_volume/run-legacy/features_table.toml",
            ],
        ) as find_file,
    ):
        artifacts = m.resolve_legacy_inference_artifacts(
            model_name="legacy_model",
            db_workspace="edvise",
            databricks_institution_name="acme",
        )

    assert find_file.call_count == 2
    assert artifacts.features_table_path.endswith("features_table.toml")


def test_set_inference_config_task_value_uses_dbutils() -> None:
    fake_dbutils = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.sdk.runtime": mock.MagicMock(dbutils=fake_dbutils)},
    ):
        m.set_inference_config_task_value("/path/to/config.toml")

    fake_dbutils.jobs.taskValues.set.assert_called_once_with(
        key="config_file_path",
        value="/path/to/config.toml",
    )


def test_set_legacy_inference_artifact_task_values() -> None:
    fake_dbutils = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.sdk.runtime": mock.MagicMock(dbutils=fake_dbutils)},
    ):
        m.set_legacy_inference_artifact_task_values(
            config_file_path="/cfg.toml",
            features_table_path="/feat.toml",
        )

    assert fake_dbutils.jobs.taskValues.set.call_count == 2
