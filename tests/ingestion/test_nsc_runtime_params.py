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


def test_helper_table_paths_follow_catalog_after_import() -> None:
    """Helpers are imported before bootstrap; table paths must still follow DB_workspace."""
    from edvise.ingestion.nsc_sftp import constants
    from edvise.ingestion.nsc_sftp.helpers import ensure_manifest_and_selected_tables

    queries: list[str] = []

    class _Spark:
        def sql(self, query: str) -> None:
            queries.append(query)

    constants.configure_nsc_catalog("staging_sst_01")
    try:
        ensure_manifest_and_selected_tables(_Spark())  # type: ignore[arg-type]
    finally:
        constants.configure_nsc_catalog(constants.DEFAULT_CATALOG_FOR_LOCAL)

    assert queries
    assert all("staging_sst_01.default." in query for query in queries)
    assert not any("dev_sst_02" in query for query in queries)


def test_configure_logging_buffers_info_into_notebook_exit() -> None:
    runtime._LOGGING_CONFIGURED = False
    runtime._LOG_BUFFER.clear()
    runtime.configure_logging()

    root = logging.getLogger()
    assert any(isinstance(h, _FlushTolerantStreamHandler) for h in root.handlers)
    assert any(isinstance(h, runtime._JobOutputBufferHandler) for h in root.handlers)

    log = runtime.get_logger("nsc.test.logging")
    log.info("hello-info")
    log.warning("hello-warn")

    captured: list[str] = []

    class _Db:
        class notebook:
            @staticmethod
            def exit(msg: str) -> None:
                captured.append(msg)

    runtime.notebook_exit(_Db(), "SUMMARY")
    assert captured and "hello-info" in captured[0]
    assert "hello-warn" in captured[0]
    assert captured[0].endswith("SUMMARY")
