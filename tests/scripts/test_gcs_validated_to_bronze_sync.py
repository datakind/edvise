"""Tests for gcs_validated_to_bronze_sync helpers and validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from google.api_core import exceptions as gax_exc
from google.api_core.exceptions import NotFound

from edvise.utils import gcs as copy_mod
from edvise.scripts import gcs_validated_to_bronze_sync as m


def _ns(**kwargs: object) -> SimpleNamespace:
    d = {
        "gcp_bucket_name": "my-bucket",
        "DB_workspace": "dev_sst_02",
        "databricks_institution_name": "acme",
        "batch_id": "",
        "gcs_source_prefix": "validated/",
        "bronze_subdir": "gcs_uploads",
        "include_blob_paths_json": "[]",
        "max_objects": 1000,
        "require_at_least_one_file": True,
        "strict_mode": "true",
    }
    d.update(kwargs)
    return SimpleNamespace(**d)


def test_parse_include_blob_paths_json_empty_list_all() -> None:
    assert m._parse_include_blob_paths_json("[]") == []
    assert m._parse_include_blob_paths_json("  ") == []


def test_parse_include_blob_paths_json_selective() -> None:
    out = m._parse_include_blob_paths_json('["validated/a.csv"]')
    assert out == ["validated/a.csv"]


def test_parse_include_invalid_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        m._parse_include_blob_paths_json("not json {")


def test_parse_include_non_list_raises() -> None:
    with pytest.raises(ValueError, match="JSON array of strings"):
        m._parse_include_blob_paths_json("1")


def test_relative_under_prefix_ok() -> None:
    assert m._relative_under_prefix("validated/a/b.csv", "validated/") == "a/b.csv"


def test_relative_under_prefix_bad() -> None:
    with pytest.raises(ValueError, match="must start with"):
        m._relative_under_prefix("other/a.csv", "validated/")


def test_sync_rejects_unsafe_db_workspace() -> None:
    args = _ns(DB_workspace="e/vil", databricks_institution_name="x")
    with pytest.raises(ValueError, match="DB_workspace"):
        m.sync_validated_to_bronze(args)


def test_sync_rejects_unsafe_databricks_institution_name() -> None:
    args = _ns(databricks_institution_name="../../evil")
    with pytest.raises(ValueError, match="databricks_institution_name"):
        m.sync_validated_to_bronze(args)


def test_is_transient_gcs_error() -> None:
    assert m._is_transient_gcs_error(TimeoutError("x")) is True
    assert m._is_transient_gcs_error(gax_exc.ServiceUnavailable("m")) is True
    assert m._is_transient_gcs_error(gax_exc.TooManyRequests("m")) is True
    assert m._is_transient_gcs_error(gax_exc.NotFound("m")) is False


def test_download_blob_retries_on_transient_then_ok(tmp_path) -> None:
    blob = mock.Mock()
    calls: list[int] = [0]

    def flaky_download(p: str) -> None:
        calls[0] += 1
        if calls[0] < 2:
            raise gax_exc.ServiceUnavailable("retry me")
        Path(p).write_text("x", encoding="utf-8")

    dest = tmp_path / "out.txt"
    blob.download_to_filename.side_effect = flaky_download

    m._download_blob_to_file(blob, str(dest))
    assert calls[0] == 2
    assert Path(dest).read_text() == "x"
    assert blob.download_to_filename.call_count == 2


def test_download_notfound_no_retry() -> None:
    blob = mock.Mock()
    blob.download_to_filename.side_effect = NotFound("nope")
    with pytest.raises(NotFound):
        m._download_blob_to_file(blob, "/tmp/should_not_matter")
    assert blob.download_to_filename.call_count == 1


def test_resolve_strict_mode() -> None:
    assert m._resolve_strict_mode("true") is True
    assert m._resolve_strict_mode("false") is False


def _volumes_to_tmp(p: str, under: Path) -> str:
    if p.startswith("/Volumes/"):
        rel = p.removeprefix("/Volumes/").lstrip("/")
        return str((under / rel).resolve())
    if p and p.startswith("dbfs:"):
        return p.replace("dbfs:/", "/dbfs/")
    return p


def test_taskvalues_set_failure_does_not_fail_sync(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path_mapper = lambda p, root=tmp_path: _volumes_to_tmp(p, root) if p else p
    monkeypatch.setattr(copy_mod, "local_fs_path", path_mapper)
    monkeypatch.setattr(m, "local_fs_path", path_mapper)

    bucket = mock.Mock()
    bobj = mock.Mock()

    def download(path: str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("1", encoding="utf-8")

    bobj.download_to_filename.side_effect = download
    bucket.blob.return_value = bobj
    client = mock.Mock()
    client.bucket = lambda n: bucket  # noqa: ARG005
    monkeypatch.setattr(m.storage, "Client", lambda: client)

    dbc = mock.Mock()
    dbc.jobs.taskValues.set.side_effect = RuntimeError("no job context")
    monkeypatch.setattr(m, "get_dbutils", lambda: dbc)
    args = _ns(
        include_blob_paths_json='["validated/x.csv"]',
        require_at_least_one_file=True,
    )
    copied, sw = m.sync_validated_to_bronze(args)
    assert sw is None
    assert copied == 1
    dbc.jobs.taskValues.set.assert_called()


def test_write_success_with_zero_copies(tmp_path) -> None:
    m._write_success_marker(
        str(tmp_path),
        copied=0,
        bucket_name="b",
        gcs_prefix="validated/",
        storage_layout="flat",
        copy_mode="all_under_prefix",
    )
    marker = tmp_path / "_SUCCESS.json"
    assert marker.is_file()
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["object_count"] == 0
