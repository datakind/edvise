"""Unit tests for ``pipelines/pdp/launchers/versioned_inference_launcher.py`` helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers import model_metadata as mm
from pipelines.pdp.launchers import versioned_inference_launcher as vil

_FIXTURE_YML = (
    Path(__file__).resolve().parent / "fixtures" / "inference_job_minimal.yml"
)


def _write_yaml_snapshot_bundle(release_dir: Path) -> None:
    release_dir.mkdir(parents=True, exist_ok=True)
    snap = release_dir / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "github_pdp_inference.yml").write_text(
        _FIXTURE_YML.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_resolve_release_dir() -> None:
    p = mm.resolve_release_dir("/vol/releases", "abc123sha")
    assert p.name == "abc123sha"
    assert str(p).endswith("abc123sha")


def test_escape_sql_string_literal() -> None:
    assert mm.escape_sql_string_literal("a'b") == "a''b"


def test_sql_select_latest_pipeline_model() -> None:
    q = mm.sql_select_latest_pipeline_model(
        "dev_sst_02", "miles_cc", "retention_into_year_2_associates"
    )
    assert "`dev_sst_02`.default.pipeline_models" in q
    assert "institution_id = 'miles_cc'" in q
    assert "model_name = 'retention_into_year_2_associates'" in q


def test_silver_training_config_path() -> None:
    p = mm.silver_training_config_path("dev_sst_02", "miles_cc", "abc123")
    assert "silver_volume" in p.parts
    assert "abc123" in p.parts
    assert p.name == "config.toml"
    assert p.parent.name == "training"


def test_pipeline_version_from_payload_json_str() -> None:
    assert mm.pipeline_version_from_payload_json_str(None) is None
    assert mm.pipeline_version_from_payload_json_str("") is None
    raw = json.dumps({"pipeline_version": "6b22fb5904c83da9d769fc4cc4d7d6d8d919520b"})
    assert (
        mm.pipeline_version_from_payload_json_str(raw)
        == "6b22fb5904c83da9d769fc4cc4d7d6d8d919520b"
    )


def test_pipeline_version_from_config_toml() -> None:
    text = 'pipeline_version = "sha_from_toml"\ninstitution_id = "x"\n'
    assert mm.pipeline_version_from_config_toml(text) == "sha_from_toml"


def test_resolve_pipeline_version_from_payload_when_config_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If silver config.toml is absent, use payload_json.pipeline_version (git SHA)."""

    def no_config(
        db_workspace: str, databricks_institution_name: str, model_run_id: str
    ) -> Path:
        return tmp_path / "nonexistent" / "config.toml"

    monkeypatch.setattr(mm, "silver_training_config_path", no_config)

    class _DF:
        def __init__(self, rows):
            self._rows = rows

        def collect(self):
            return self._rows

    class _Spark:
        def __init__(self, rows):
            self.rows = rows

        def sql(self, _q):
            return _DF(self.rows)

    payload = json.dumps(
        {"pipeline_version": "6b22fb5904c83da9d769fc4cc4d7d6d8d919520b"}
    )
    spark = _Spark(
        [{"model_run_id": "9e5494d8774c4f62917d4c569aa0ce95", "payload_json": payload}]
    )
    out = mm.resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace="dev_sst_02",
        databricks_institution_name="san_jose_state_uni_pdp",
        model_name="graduation_in_4y_ft_checkpoint_4_core_terms",
    )
    assert out == (
        "9e5494d8774c4f62917d4c569aa0ce95",
        "6b22fb5904c83da9d769fc4cc4d7d6d8d919520b",
    )


def test_resolve_pipeline_version_prefers_config_over_payload_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('pipeline_version = "from_config_toml"\n', encoding="utf-8")

    def fake_silver_path(
        db_workspace: str, databricks_institution_name: str, model_run_id: str
    ) -> Path:
        return cfg

    monkeypatch.setattr(mm, "silver_training_config_path", fake_silver_path)

    class _DF:
        def __init__(self, rows):
            self._rows = rows

        def collect(self):
            return self._rows

    class _Spark:
        def __init__(self, rows):
            self.rows = rows

        def sql(self, _q):
            return _DF(self.rows)

    payload = json.dumps({"pipeline_version": "sha_from_payload_only"})
    spark = _Spark([{"model_run_id": "mr1", "payload_json": payload}])
    out = mm.resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace="dev_sst_02",
        databricks_institution_name="miles_cc",
        model_name="retention_into_year_2_associates",
    )
    assert out == ("mr1", "from_config_toml")


def test_resolve_model_run_fallback_config_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('pipeline_version = "from_config"\n', encoding="utf-8")

    def fake_silver_path(
        db_workspace: str, databricks_institution_name: str, model_run_id: str
    ) -> Path:
        return cfg

    monkeypatch.setattr(mm, "silver_training_config_path", fake_silver_path)

    class _DF:
        def __init__(self, rows):
            self._rows = rows

        def collect(self):
            return self._rows

    class _Spark:
        def __init__(self, rows):
            self.rows = rows

        def sql(self, _q):
            return _DF(self.rows)

    spark = _Spark([{"model_run_id": "mr1", "payload_json": "{}"}])
    out = mm.resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace="dev_sst_02",
        databricks_institution_name="miles_cc",
        model_name="retention_into_year_2_associates",
    )
    assert out == ("mr1", "from_config")


def test_normalize_uc_model_short_name() -> None:
    assert (
        mm.normalize_uc_model_short_name(
            "dev_sst_02.midway_uni_gold.graduation_in_4y",
            workspace="dev_sst_02",
            institution="midway_uni",
        )
        == "graduation_in_4y"
    )
    assert mm.normalize_uc_model_short_name("short_only") == "short_only"


def test_resolve_model_run_id_from_uc_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    class _Version:
        def __init__(self, version: str, run_id: str) -> None:
            self.version = version
            self.run_id = run_id

    class _Client:
        def search_model_versions(self, filter_string: str) -> list[_Version]:
            assert filter_string == (
                "name='dev_sst_02.midway_uni_gold.graduation_in_4y_ft_4y_pt_checkpoint_2_core_terms'"
            )
            return [
                _Version("1", "run-old"),
                _Version("3", "run-latest"),
            ]

    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = lambda registry_uri="databricks-uc": _Client()
    mlflow_mod = types.ModuleType("mlflow")
    mlflow_mod.tracking = tracking
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)

    out = mm.resolve_model_run_id_from_uc_registry(
        db_workspace="dev_sst_02",
        databricks_institution_name="midway_uni",
        model_name="graduation_in_4y_ft_4y_pt_checkpoint_2_core_terms",
    )
    assert out == "run-latest"


def test_resolve_model_run_no_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mm,
        "resolve_model_run_id_from_uc_registry",
        lambda **_: None,
    )

    class _DF:
        def collect(self):
            return []

    class _Spark:
        def sql(self, _q):
            return _DF()

    assert (
        mm.resolve_model_run_and_pipeline_version(
            spark=_Spark(),
            db_workspace="dev_sst_02",
            databricks_institution_name="miles_cc",
            model_name="missing_model",
        )
        is None
    )


def test_resolve_model_run_from_uc_silver_when_no_pipeline_models_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'pipeline_version = "sha_from_silver_via_uc"\n',
        encoding="utf-8",
    )

    def fake_silver_path(
        db_workspace: str, databricks_institution_name: str, model_run_id: str
    ) -> Path:
        assert model_run_id == "uc-run-abc"
        return cfg

    monkeypatch.setattr(mm, "silver_training_config_path", fake_silver_path)
    monkeypatch.setattr(
        mm,
        "resolve_model_run_id_from_uc_registry",
        lambda **_: "uc-run-abc",
    )

    class _DF:
        def collect(self):
            return []

    class _Spark:
        def sql(self, _q):
            return _DF()

    out = mm.resolve_model_run_and_pipeline_version(
        spark=_Spark(),
        db_workspace="dev_sst_02",
        databricks_institution_name="midway_uni",
        model_name="graduation_in_4y_ft_4y_pt_checkpoint_2_core_terms",
    )
    assert out == ("uc-run-abc", "sha_from_silver_via_uc")


def test_resolve_model_run_explicit_override_skips_lookups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('pipeline_version = "override_sha"\n', encoding="utf-8")

    def fake_silver_path(
        db_workspace: str, databricks_institution_name: str, model_run_id: str
    ) -> Path:
        return cfg

    monkeypatch.setattr(mm, "silver_training_config_path", fake_silver_path)

    def fail_sql(_q):
        raise AssertionError("pipeline_models should not be queried")

    class _Spark:
        def sql(self, q):
            return fail_sql(q)

    out = mm.resolve_model_run_and_pipeline_version(
        spark=_Spark(),
        db_workspace="dev_sst_02",
        databricks_institution_name="midway_uni",
        model_name="any_model",
        model_run_id_override="explicit-run-id",
    )
    assert out == ("explicit-run-id", "override_sha")


def test_main_ok_yaml_snapshot_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rel = tmp_path / "v1"
    _write_yaml_snapshot_bundle(rel)

    monkeypatch.setattr(vil, "get_spark_session", lambda: object())
    monkeypatch.setattr(
        vil,
        "resolve_model_run_and_pipeline_version",
        lambda **_: ("resolved-mr", "v1"),
    )
    # Bundle metadata maps DBR 15.4 → Python 3.11; CI may run 3.10.
    monkeypatch.setattr(
        vil,
        "check_runtime_bundle_compatibility",
        lambda *a, **k: (True, ""),
    )

    argv = [
        "--databricks_institution_name",
        "miles_cc",
        "--model_name",
        "retention_into_year_2_associates",
        "--DB_workspace",
        "dev_sst_02",
        "--release_base_path",
        str(tmp_path),
    ]
    assert vil.main(argv) == 0


def test_main_requires_institution_model_workspace() -> None:
    assert vil.main(["--DB_workspace", "dev_sst_02"]) == 1


def test_parse_python_xy() -> None:
    assert vil.parse_python_xy("3.11") == (3, 11)
    assert vil.parse_python_xy("nope") is None


def test_check_runtime_bundle_execution_mode_dab() -> None:
    ok, msg = vil.check_runtime_bundle_compatibility(
        {"execution_mode": "dab"},
        spark=type("S", (), {"version": "3.5.2"})(),
    )
    assert ok is False
    assert "dab" in msg.lower()


def test_check_runtime_bundle_dbr_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3.x-cpu-ml-scala2.12")
    ok, msg = vil.check_runtime_bundle_compatibility(
        {"required_runtime": {"databricks_runtime": "15.4.x-cpu-ml-scala2.12"}},
        spark=type("S", (), {"version": "3.5.2"})(),
    )
    assert ok is False
    assert "15.4" in msg


def test_databricks_runtime_compatible_short_cluster_version() -> None:
    assert vil.databricks_runtime_compatible("15.4.x-cpu-ml-scala2.12", "15.4")
    assert vil.databricks_runtime_compatible(
        "15.4.x-cpu-ml-scala2.12", "15.4.x-cpu-ml-scala2.12"
    )
    assert not vil.databricks_runtime_compatible("15.4.x-cpu-ml-scala2.12", "14.3")


def test_check_runtime_bundle_dbr_short_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    ok, msg = vil.check_runtime_bundle_compatibility(
        {"required_runtime": {"databricks_runtime": "15.4.x-cpu-ml-scala2.12"}},
        spark=type("S", (), {"version": "3.5.2"})(),
    )
    assert ok is True
    assert msg == ""
