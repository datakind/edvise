from pathlib import Path

import pytest

from edvise.scripts.legacy_preprocessing import (
    DEFAULT_FEATURES_TABLE_NAME,
    DEFAULT_LEGACY_CONFIG_BASENAME,
    legacy_training_inputs_uc_dir,
    resolve_legacy_training_toml_paths,
)


def test_legacy_training_inputs_uc_dir():
    path = legacy_training_inputs_uc_dir("dev_sst_02", "john_jay_col")
    assert path == Path(
        "/Volumes/dev_sst_02/john_jay_col_bronze/bronze_volume/training_inputs"
    )


@pytest.fixture
def training_inputs_dir(tmp_path: Path, monkeypatch):
    base = tmp_path / "training_inputs"
    base.mkdir(parents=True, exist_ok=True)

    def fake_uc_dir(db_workspace: str, institution_id: str) -> Path:
        return base

    monkeypatch.setattr(
        "edvise.scripts.legacy_preprocessing.legacy_training_inputs_uc_dir",
        fake_uc_dir,
    )
    return base


def test_resolve_legacy_training_toml_paths(training_inputs_dir: Path):
    training_inputs_dir.joinpath(DEFAULT_LEGACY_CONFIG_BASENAME).write_text(
        "institution_id = 'x'\n", encoding="utf-8"
    )
    training_inputs_dir.joinpath(DEFAULT_FEATURES_TABLE_NAME).write_text(
        "feat = { name = 'Feat' }\n", encoding="utf-8"
    )

    cfg, feat = resolve_legacy_training_toml_paths("dev_sst_02", "john_jay_col")
    assert cfg.endswith(DEFAULT_LEGACY_CONFIG_BASENAME)
    assert feat.endswith(DEFAULT_FEATURES_TABLE_NAME)


def test_resolve_legacy_training_toml_paths_missing_config(training_inputs_dir: Path):
    training_inputs_dir.joinpath(DEFAULT_FEATURES_TABLE_NAME).write_text(
        "x = { name = 'X' }\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Training config TOML not found"):
        resolve_legacy_training_toml_paths("dev_sst_02", "john_jay_col")


def test_resolve_legacy_training_toml_paths_missing_features(training_inputs_dir: Path):
    training_inputs_dir.joinpath(DEFAULT_LEGACY_CONFIG_BASENAME).write_text(
        "x = 1\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Features table TOML not found"):
        resolve_legacy_training_toml_paths("dev_sst_02", "john_jay_col")


def test_resolve_legacy_training_toml_paths_custom_config_name(
    training_inputs_dir: Path,
):
    training_inputs_dir.joinpath("config_primary_n0.toml").write_text(
        "x = 1\n", encoding="utf-8"
    )
    training_inputs_dir.joinpath(DEFAULT_FEATURES_TABLE_NAME).write_text(
        "a = { name = 'A' }\n", encoding="utf-8"
    )
    cfg, feat = resolve_legacy_training_toml_paths(
        "dev_sst_02",
        "john_jay_col",
        config_file_name="config_primary_n0.toml",
    )
    assert cfg.endswith("config_primary_n0.toml")
    assert feat.endswith(DEFAULT_FEATURES_TABLE_NAME)


def test_resolve_legacy_training_toml_paths_custom_features_name(
    training_inputs_dir: Path,
):
    training_inputs_dir.joinpath(DEFAULT_LEGACY_CONFIG_BASENAME).write_text(
        "x = 1\n", encoding="utf-8"
    )
    training_inputs_dir.joinpath("custom_features.toml").write_text(
        "a = { name = 'A' }\n", encoding="utf-8"
    )
    cfg, feat = resolve_legacy_training_toml_paths(
        "dev_sst_02",
        "john_jay_col",
        features_table_name="custom_features.toml",
    )
    assert cfg.endswith(DEFAULT_LEGACY_CONFIG_BASENAME)
    assert feat.endswith("custom_features.toml")
