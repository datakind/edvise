from pathlib import Path

import pytest
import time

from edvise.dataio.path_management import (
    find_predict_file_in_directory,
    legacy_bronze_gcs_uploads_dir,
    normalize_predict_file_match_text,
    predict_file_keywords,
    resolve_legacy_bronze_predict_file,
)
from edvise.scripts.legacy_preprocessing import materialize_legacy_bronze_predict_paths


def test_legacy_bronze_gcs_uploads_dir():
    assert legacy_bronze_gcs_uploads_dir("dev_sst_02", "john_jay_col") == (
        "/Volumes/dev_sst_02/john_jay_col_bronze/bronze_volume/gcs_uploads"
    )


def test_find_predict_file_in_directory_keyword_picks_newest(tmp_path: Path):
    older = tmp_path / "cohort_inference_2024.csv"
    newer = tmp_path / "cohort_inference_2025.csv"
    older.write_text("old", encoding="utf-8")
    newer.write_text("new", encoding="utf-8")
    time.sleep(0.01)
    newer.touch()

    resolved = find_predict_file_in_directory(
        str(tmp_path),
        keyword="cohort_inference",
        label="raw_student",
    )
    assert resolved == str(newer)


def test_find_predict_file_in_directory_no_match(tmp_path: Path):
    tmp_path.joinpath("semester.csv").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no matching file"):
        find_predict_file_in_directory(
            str(tmp_path),
            keyword="cohort",
            label="raw_student",
        )


def test_resolve_legacy_bronze_predict_file_uses_explicit_path(tmp_path: Path):
    csv = tmp_path / "explicit.csv"
    csv.write_text("data", encoding="utf-8")
    ds = {"predict_file_path": str(csv)}
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_student",
        db_workspace="dev_sst_02",
        institution_id="inst",
    ) == str(csv)


def test_resolve_legacy_bronze_predict_file_discovers_via_keyword(tmp_path: Path):
    csv = tmp_path / "transfer_advisement_fall.csv"
    csv.write_text("data", encoding="utf-8")
    ds = {
        "predict_file_path": str(tmp_path / "missing.csv"),
        "predict_file_keyword": "transfer",
        "train_file_path": str(tmp_path / "train.csv"),
    }
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_student",
        db_workspace="dev_sst_02",
        institution_id="inst",
    ) == str(csv)


def test_resolve_legacy_bronze_predict_file_prefers_keyword_over_stale_explicit(
    tmp_path: Path,
):
    stale = tmp_path / "transfer_advisement_old.csv"
    fresh = tmp_path / "transfer_advisement_fall2026.csv"
    stale.write_text("old", encoding="utf-8")
    fresh.write_text("new", encoding="utf-8")
    time.sleep(0.01)
    fresh.touch()
    ds = {
        "predict_file_path": str(stale),
        "predict_file_keyword": "transfer_advisement",
        "train_file_path": str(tmp_path / "train.csv"),
    }
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_cuny_transfer",
        db_workspace="dev_sst_02",
        institution_id="john_jay_col",
    ) == str(fresh)


def test_resolve_legacy_bronze_predict_file_prefers_batch_dir_over_top_level(
    tmp_path: Path,
):
    top_level_dir = tmp_path / "gcs_uploads"
    top_level_dir.mkdir()
    stale = top_level_dir / "transfer_advisement_old.csv"
    stale.write_text("old", encoding="utf-8")

    batch_dir = tmp_path / "gcs_uploads" / "batch123"
    batch_dir.mkdir()
    fresh = batch_dir / "transfer_advisement_batch.csv"
    fresh.write_text("new", encoding="utf-8")

    ds = {"predict_file_keyword": "transfer_advisement"}
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_cuny_transfer",
        db_workspace="dev_sst_02",
        institution_id="john_jay_col",
        bronze_batch_dir=str(batch_dir),
    ) == str(fresh)


def test_resolve_legacy_bronze_predict_file_batch_dir_beats_newer_fallback(
    tmp_path: Path,
):
    top_level_dir = tmp_path / "gcs_uploads"
    top_level_dir.mkdir()
    batch_dir = top_level_dir / "batch123"
    batch_dir.mkdir()

    in_batch = batch_dir / "transfer_advisement_batch.csv"
    in_batch.write_text("batch", encoding="utf-8")
    newer_outside_batch = top_level_dir / "transfer_advisement_newer.csv"
    newer_outside_batch.write_text("newer", encoding="utf-8")
    time.sleep(0.01)
    newer_outside_batch.touch()

    ds = {"predict_file_keyword": "transfer_advisement"}
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_cuny_transfer",
        db_workspace="dev_sst_02",
        institution_id="john_jay_col",
        bronze_batch_dir=str(batch_dir),
    ) == str(in_batch)


def test_resolve_legacy_bronze_predict_file_matches_any_keyword_alias(tmp_path: Path):
    renamed = tmp_path / "DEIDENTIFIED ir_trn_advsmnt_model_jjc01 2006 08 24.csv"
    renamed.write_text("data", encoding="utf-8")

    ds = {
        "predict_file_keyword": ["transfer_advisement_model_data", "ir_trn_advsmnt"],
        "train_file_path": str(tmp_path / "train.csv"),
    }
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_cuny_transfer",
        db_workspace="dev_sst_02",
        institution_id="john_jay_col",
    ) == str(renamed)


def test_resolve_legacy_bronze_predict_file_ignores_separator_spelling(tmp_path: Path):
    spaced = tmp_path / "DE-ID CU SR DEG AWARDED DEPT 08_2026.csv"
    spaced.write_text("data", encoding="utf-8")

    ds = {
        "predict_file_keyword": "cu_sr_deg_awarded_dept",
        "train_file_path": str(tmp_path / "train.csv"),
    }
    assert resolve_legacy_bronze_predict_file(
        ds,
        dataset_key="raw_degrees",
        db_workspace="dev_sst_02",
        institution_id="john_jay_col",
    ) == str(spaced)


def test_resolve_legacy_bronze_predict_file_raises_when_no_keyword_matches(
    tmp_path: Path,
):
    tmp_path.joinpath("unrelated_extract.csv").write_text("x", encoding="utf-8")
    ds = {
        "predict_file_keyword": ["transfer_advisement", "ir_trn_advsmnt"],
        "train_file_path": str(tmp_path / "train.csv"),
    }
    with pytest.raises(FileNotFoundError, match="no file matching"):
        resolve_legacy_bronze_predict_file(
            ds,
            dataset_key="raw_cuny_transfer",
            db_workspace="dev_sst_02",
            institution_id="john_jay_col",
        )


def test_find_predict_file_in_directory_accepts_keyword_list(tmp_path: Path):
    csv = tmp_path / "ir_trn_advsmnt_model_jjc01.csv"
    csv.write_text("data", encoding="utf-8")
    resolved = find_predict_file_in_directory(
        str(tmp_path),
        keyword=["transfer_advisement_model_data", "ir_trn_advsmnt"],
        label="raw_cuny_transfer",
    )
    assert resolved == str(csv)


def test_normalize_predict_file_match_text():
    assert (
        normalize_predict_file_match_text("DEIDENTIFIED ir_trn-advsmnt (Fall 26).csv")
        == "deidentified_ir_trn_advsmnt_fall_26_csv"
    )


def test_predict_file_keywords_normalizes_and_dedupes():
    assert predict_file_keywords({}) == []
    assert predict_file_keywords(
        {"predict_file_keyword": "  Transfer Advisement "}
    ) == ["transfer_advisement"]
    assert predict_file_keywords(
        {"predict_file_keyword": ["A-B", "a_b", "", "  ", "c"]}
    ) == ["a_b", "c"]


def test_materialize_legacy_bronze_predict_paths_writes_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    upload_dir = tmp_path / "gcs_uploads"
    upload_dir.mkdir()
    cohort = upload_dir / "cohort_inference_batch.csv"
    cohort.write_text("cohort", encoding="utf-8")

    monkeypatch.setattr(
        "edvise.dataio.path_management.legacy_bronze_gcs_uploads_dir",
        lambda _ws, _inst: str(upload_dir),
    )

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
institution_id = "inst"
institution_name = "Inst"

[datasets.bronze.raw_student]
train_file_path = "/tmp/train.csv"
file_path = "/tmp/train.csv"
predict_file_keyword = "cohort_inference"

[datasets.silver.modeling]
train_table_path = "cat.inst_silver.modeling"

[datasets.silver.model_features]
predict_table_path = "cat.inst_silver.model_features"
""".strip(),
        encoding="utf-8",
    )

    out = materialize_legacy_bronze_predict_paths(str(cfg_path), "dev_sst_02")
    assert out != str(cfg_path)
    text = Path(out).read_text(encoding="utf-8")
    assert str(cohort) in text
    assert "predict_file_path" in text
    assert "file_path" in text


def test_dataset_config_accepts_predict_keyword_only():
    from edvise.configs import legacy

    ds = legacy.DatasetConfig.model_validate({"predict_file_keyword": "cohort"})
    assert ds.predict_file_keyword == "cohort"


def test_dataset_config_accepts_predict_keyword_list():
    from edvise.configs import legacy

    ds = legacy.DatasetConfig.model_validate(
        {"predict_file_keyword": ["cohort", "student_file"]}
    )
    assert ds.predict_file_keyword == ["cohort", "student_file"]


def test_dataset_config_rejects_blank_predict_keyword_list():
    import pydantic

    from edvise.configs import legacy

    with pytest.raises(pydantic.ValidationError, match="At least one dataset path"):
        legacy.DatasetConfig.model_validate({"predict_file_keyword": ["", "  "]})
