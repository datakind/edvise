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


def _write_test_release_bundle(release_dir: Path, *, wheel_name: str = "fake.whl") -> None:
    release_dir.mkdir(parents=True, exist_ok=True)
    snap = release_dir / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "github_pdp_inference.yml").write_text(
        _FIXTURE_YML.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (release_dir / wheel_name).write_bytes(b"")
    (release_dir / "release.json").write_text(
        json.dumps({"wheel": wheel_name}),
        encoding="utf-8",
    )


def test_resolve_release_dir() -> None:
    p = vil.resolve_release_dir("/vol/releases", "abc123sha")
    assert p.name == "abc123sha"
    assert str(p).endswith("abc123sha")


def test_build_payload_dict_merges_extra() -> None:
    release = {"pipeline_version": "v1", "wheel": "w.whl", "entrypoint": "m"}
    extra = {"note": "hello"}
    out = vil.build_payload_dict("run-1", "v1", release, extra)
    assert out["model_run_id"] == "run-1"
    assert out["pipeline_version"] == "v1"
    assert out["release"] == release
    assert out["note"] == "hello"


def test_build_payload_dict_no_extra() -> None:
    release = {"pipeline_version": "v1", "wheel": "w.whl", "entrypoint": "m"}
    out = vil.build_payload_dict("r", "v1", release, None)
    assert out["model_run_id"] == "r"


def test_build_payload_dict_includes_institution_and_model() -> None:
    release = {"pipeline_version": "v1", "wheel": "w.whl", "entrypoint": "m"}
    out = vil.build_payload_dict(
        "r",
        "v1",
        release,
        None,
        databricks_institution_name="miles_cc",
        model_name="retention_into_year_2_associates",
    )
    assert out["databricks_institution_name"] == "miles_cc"
    assert out["model_name"] == "retention_into_year_2_associates"


def test_escape_sql_string_literal() -> None:
    assert vil.escape_sql_string_literal("a'b") == "a''b"


def test_sql_select_latest_pipeline_model() -> None:
    q = vil.sql_select_latest_pipeline_model(
        "dev_sst_02", "miles_cc", "retention_into_year_2_associates"
    )
    assert "`dev_sst_02`.default.pipeline_models" in q
    assert "institution_id = 'miles_cc'" in q
    assert "model_name = 'retention_into_year_2_associates'" in q


def test_silver_training_config_path() -> None:
    p = vil.silver_training_config_path("dev_sst_02", "miles_cc", "abc123")
    assert "silver_volume" in p.parts
    assert "abc123" in p.parts
    assert p.name == "config.toml"
    assert p.parent.name == "training"


def test_pipeline_version_from_payload_json_str() -> None:
    assert vil.pipeline_version_from_payload_json_str(None) is None
    assert vil.pipeline_version_from_payload_json_str("") is None
    raw = json.dumps({"pipeline_version": "6b22fb5904c83da9d769fc4cc4d7d6d8d919520b"})
    assert (
        vil.pipeline_version_from_payload_json_str(raw)
        == "6b22fb5904c83da9d769fc4cc4d7d6d8d919520b"
    )


def test_pipeline_version_from_config_toml() -> None:
    text = 'pipeline_version = "sha_from_toml"\ninstitution_id = "x"\n'
    assert vil.pipeline_version_from_config_toml(text) == "sha_from_toml"


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
    out = vil.resolve_model_run_and_pipeline_version(
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
    out = vil.resolve_model_run_and_pipeline_version(
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
    out = vil.resolve_model_run_and_pipeline_version(
        spark=spark,
        db_workspace="dev_sst_02",
        databricks_institution_name="miles_cc",
        model_name="retention_into_year_2_associates",
    )
    assert out == ("mr1", "from_config")


def test_resolve_model_run_no_rows() -> None:
    class _DF:
        def collect(self):
            return []

    class _Spark:
        def sql(self, _q):
            return _DF()

    assert (
        vil.resolve_model_run_and_pipeline_version(
            spark=_Spark(),
            db_workspace="dev_sst_02",
            databricks_institution_name="miles_cc",
            model_name="missing_model",
        )
        is None
    )


def test_resolve_wheel_path() -> None:
    p = vil.resolve_wheel_path(Path("/releases/v9"), "pkg.whl")
    assert p.name == "pkg.whl"


def test_pip_install_wheel_command() -> None:
    cmd = vil.pip_install_wheel_command("/usr/bin/python3", "/w/x.whl")
    assert cmd == [
        "/usr/bin/python3",
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "/w/x.whl",
    ]


def test_verify_edvise_import_command() -> None:
    cmd = vil.verify_edvise_import_command("/py")
    assert cmd[:2] == ["/py", "-c"]
    assert "edvise.__file__" in cmd[2]


def test_entrypoint_command() -> None:
    cmd = vil.entrypoint_command("/py", "edvise.runtime.inference_driver", "/tmp/p.json")
    assert cmd == [
        "/py",
        "-m",
        "edvise.runtime.inference_driver",
        "--payload",
        "/tmp/p.json",
    ]


def test_main_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rel = tmp_path / "v1"
    _write_test_release_bundle(rel)

    monkeypatch.setattr(vil, "get_spark_session", lambda: object())
    monkeypatch.setattr(
        vil,
        "resolve_model_run_and_pipeline_version",
        lambda **_: ("resolved-mr", "v1"),
    )

    recorded: list[tuple[str, list[str]]] = []

    def fake_run(cmd: list[str], *, label: str, logger=None) -> int:
        recorded.append((label, list(cmd)))
        return 0

    def fake_write(
        payload: dict,
        base_dir: Path | None = None,
    ) -> Path:
        out = tmp_path / "payload.json"
        out.write_text(json.dumps(payload), encoding="utf-8")
        return out

    monkeypatch.setattr(vil, "run_logged_subprocess", fake_run)
    monkeypatch.setattr(vil, "write_payload_file", fake_write)

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
    assert [r[0] for r in recorded] == [
        "pip_install_wheel",
        "verify_edvise_import",
        "versioned_entrypoint",
    ]
    assert "fake.whl" in recorded[0][1][-1]
    assert recorded[2][1][:2] == [sys.executable, "-m"]
    assert recorded[2][1][2] == "edvise.runtime.inference_driver"


def test_main_pip_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rel = tmp_path / "v1"
    _write_test_release_bundle(rel)

    monkeypatch.setattr(vil, "get_spark_session", lambda: object())
    monkeypatch.setattr(
        vil,
        "resolve_model_run_and_pipeline_version",
        lambda **_: ("mr", "v1"),
    )

    def fake_run(cmd: list[str], *, label: str, logger=None) -> int:
        if label == "pip_install_wheel":
            return 1
        return 0

    monkeypatch.setattr(vil, "run_logged_subprocess", fake_run)

    argv = [
        "--databricks_institution_name",
        "miles_cc",
        "--model_name",
        "m",
        "--DB_workspace",
        "dev_sst_02",
        "--release_base_path",
        str(tmp_path),
    ]
    assert vil.main(argv) == 1


def test_main_missing_wheel_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rel = tmp_path / "v1"
    snap = rel / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    (snap / "github_pdp_inference.yml").write_text(
        _FIXTURE_YML.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (rel / "release.json").write_text(
        json.dumps({"wheel": "missing.whl"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(vil, "get_spark_session", lambda: object())
    monkeypatch.setattr(
        vil,
        "resolve_model_run_and_pipeline_version",
        lambda **_: ("mr", "v1"),
    )

    argv = [
        "--databricks_institution_name",
        "miles_cc",
        "--model_name",
        "m",
        "--DB_workspace",
        "dev_sst_02",
        "--release_base_path",
        str(tmp_path),
    ]
    assert vil.main(argv) == 1


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
    assert vil.databricks_runtime_compatible(
        "15.4.x-cpu-ml-scala2.12", "15.4"
    )
    assert vil.databricks_runtime_compatible(
        "15.4.x-cpu-ml-scala2.12", "15.4.x-cpu-ml-scala2.12"
    )
    assert not vil.databricks_runtime_compatible(
        "15.4.x-cpu-ml-scala2.12", "14.3"
    )


def test_check_runtime_bundle_dbr_short_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    ok, msg = vil.check_runtime_bundle_compatibility(
        {"required_runtime": {"databricks_runtime": "15.4.x-cpu-ml-scala2.12"}},
        spark=type("S", (), {"version": "3.5.2"})(),
    )
    assert ok is True
    assert msg == ""


def test_validate_required_payload_fields() -> None:
    eff = {"required_payload_fields": ["model_run_id", "extra_field"]}
    bad_ok, bad_msg = vil.validate_required_payload_fields(
        eff, {"model_run_id": "x"}
    )
    assert bad_ok is False
    assert "extra_field" in bad_msg
    good_ok, _ = vil.validate_required_payload_fields(
        eff, {"model_run_id": "x", "extra_field": 1}
    )
    assert good_ok is True


def test_run_logged_subprocess_uses_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    def fake_run(*_a, **_kw):
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(vil.subprocess, "run", fake_run)
    assert vil.run_logged_subprocess([sys.executable, "-c", "print(1)"], label="t") == 0
