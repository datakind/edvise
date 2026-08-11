import pytest

from edvise.ingestion.nsc_sftp import runtime


def test_job_param_bool_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "job_param", lambda name, default="", **_: "true")
    assert runtime.job_param_bool("force_reingest") is True


def test_job_param_bool_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "job_param", lambda name, default="", **_: "false")
    assert runtime.job_param_bool("force_reingest") is False


def test_job_param_bool_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "job_param", lambda name, default="", **_: "maybe")
    with pytest.raises(ValueError, match="Invalid boolean"):
        runtime.job_param_bool("force_reingest")
