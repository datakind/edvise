import logging

import pytest

from edvise.ingestion.nsc_sftp import runtime
from edvise.shared.logger import _FlushTolerantStreamHandler


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


def test_configure_logging_enables_info_on_stdout() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    warn_handler = logging.StreamHandler()
    warn_handler.setLevel(logging.WARNING)
    root.addHandler(warn_handler)
    root.setLevel(logging.WARNING)

    runtime._LOGGING_CONFIGURED = False
    runtime.configure_logging()

    assert root.level == logging.INFO
    assert any(isinstance(h, _FlushTolerantStreamHandler) for h in root.handlers)
    assert runtime.get_logger("nsc.test.logging").isEnabledFor(logging.INFO)
