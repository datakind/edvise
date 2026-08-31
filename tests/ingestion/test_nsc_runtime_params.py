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


def test_configure_logging_buffers_info_into_notebook_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime._LOGGING_CONFIGURED = False
    runtime._LOG_BUFFER.clear()
    runtime.configure_logging()

    log = runtime.get_logger("nsc.test.logging")
    log.info("hello-info")
    log.warning("hello-warn")

    captured = []

    class _Db:
        class notebook:
            @staticmethod
            def exit(msg: str) -> None:
                captured.append(msg)

    runtime.notebook_exit(_Db(), "SUMMARY")
    assert captured and "hello-info" in captured[0]
    assert "hello-warn" in captured[0]
    assert captured[0].endswith("SUMMARY")
    out = capsys.readouterr().out
    assert "hello-info" in out
    assert "hello-warn" in out
