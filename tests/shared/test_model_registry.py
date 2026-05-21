"""Tests for edvise.shared.model_registry."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from edvise.shared import model_registry as mr


def _registered(name: str, created_ms: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        creation_timestamp=created_ms,
        last_updated_timestamp=created_ms,
    )


def _version(version: int, run_id: str, created_ms: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        version=str(version),
        run_id=run_id,
        creation_timestamp=created_ms,
    )


class _PagedModels:
    def __init__(self, models: list[SimpleNamespace], token: str | None = None):
        self.registered_models = models
        self.token = token


class FakeMlflowClient:
    def __init__(
        self,
        models: list[SimpleNamespace] | None = None,
        versions_by_name: dict[str, list[SimpleNamespace]] | None = None,
        runs_by_id: dict[str, SimpleNamespace] | None = None,
        *,
        page_size: int | None = None,
    ):
        self._models = models or []
        self._versions_by_name = versions_by_name or {}
        self._runs_by_id = runs_by_id or {}
        self._page_size = page_size

    def search_registered_models(self, max_results=1000, page_token=None):
        if self._page_size is None:
            return self._models
        start = 0 if not page_token else int(page_token)
        end = start + min(max_results, self._page_size)
        chunk = self._models[start:end]
        next_token = str(end) if end < len(self._models) else None
        return _PagedModels(chunk, next_token)

    def search_model_versions(self, filter_string: str):
        # filter_string is name='catalog.schema.model'
        name = filter_string.split("'", 2)[1]
        return self._versions_by_name.get(name, [])

    def get_run(self, run_id: str):
        if run_id not in self._runs_by_id:
            raise KeyError(run_id)
        return self._runs_by_id[run_id]


def test_pick_newest_registered_model_by_created_at():
    models = [
        _registered("cat.school_gold.old_model", 100),
        _registered("cat.school_gold.new_model", 500),
    ]
    picked = mr.pick_newest_registered_model_by_created_at(models)
    assert picked.name == "cat.school_gold.new_model"


def test_pick_latest_model_version():
    versions = [_version(1, "run_a"), _version(3, "run_c"), _version(2, "run_b")]
    picked = mr.pick_latest_model_version(versions)
    assert picked.run_id == "run_c"
    assert picked.version == "3"


def test_get_latest_registered_model_run_id_newest_model():
    client = FakeMlflowClient(
        models=[
            _registered("staging.jf_drake_state_cc_gold.model_a", 100),
            _registered("staging.jf_drake_state_cc_gold.model_b", 900),
        ],
        versions_by_name={
            "staging.jf_drake_state_cc_gold.model_b": [
                _version(1, "run_old"),
                _version(2, "e39e2308e9da4a7aaefc71b7c5d97e61"),
            ],
        },
    )
    run_id, full_name, version = mr.get_latest_registered_model_run_id(
        client,
        catalog="staging",
        institution_id="jf_drake_state_cc",
    )
    assert full_name == "staging.jf_drake_state_cc_gold.model_b"
    assert run_id == "e39e2308e9da4a7aaefc71b7c5d97e61"
    assert version == "2"


def test_get_latest_registered_model_run_id_named_model():
    client = FakeMlflowClient(
        models=[_registered("staging.school_gold.retention_model", 1)],
        versions_by_name={
            "staging.school_gold.retention_model": [_version(5, "specific_run")],
        },
    )
    run_id, full_name, version = mr.get_latest_registered_model_run_id(
        client,
        catalog="staging",
        institution_id="school",
        model_name="retention_model",
    )
    assert run_id == "specific_run"
    assert full_name == "staging.school_gold.retention_model"
    assert version == "5"


def test_find_model_card_pdf_under_run_dir(tmp_path):
    run_dir = tmp_path / "e39e2308"
    run_dir.mkdir()
    (run_dir / "notes.txt").write_text("x", encoding="utf-8")
    (run_dir / "other.pdf").write_text("%PDF", encoding="utf-8")
    card = run_dir / "model-card-retention.pdf"
    card.write_text("%PDF", encoding="utf-8")

    path = mr.find_model_card_pdf_under_run_dir(str(run_dir))
    assert path.endswith("model-card-retention.pdf")


def test_find_model_card_pdf_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mr.find_model_card_pdf_under_run_dir(str(tmp_path / "missing"))


def test_search_institution_registered_models_paginates_uc_registry():
    """Models not on the first registry page must still be found (e.g. valencia_col)."""
    page1 = [_registered(f"staging.other_gold.model_{i}", i) for i in range(3)]
    valencia = _registered(
        "staging.valencia_col_gold.valencia_col_15_creds_in_2_years_first_term",
        999,
    )
    client = FakeMlflowClient(models=page1 + [valencia], page_size=3)
    found = mr.search_institution_registered_models(
        client,
        catalog="staging",
        institution_id="valencia_col",
    )
    assert [m.name for m in found] == [valencia.name]


def test_gold_model_cards_run_dir():
    assert mr.gold_model_cards_run_dir("staging_sst_01", "jf_drake_state_cc", "abc") == (
        "/Volumes/staging_sst_01/jf_drake_state_cc_gold/gold_volume/model_cards/abc"
    )


def test_run_start_yyyymmdd():
    # Mar 02, 2026 15:49 UTC
    start_ms = int(
        datetime(2026, 3, 2, 15, 49, tzinfo=timezone.utc).timestamp() * 1000
    )
    client = FakeMlflowClient(
        runs_by_id={
            "run_1": SimpleNamespace(info=SimpleNamespace(start_time=start_ms)),
        }
    )
    assert mr.run_start_yyyymmdd(client, "run_1") == "20260302"


def test_model_card_copy_basename():
    assert (
        mr.model_card_copy_basename(
            "/Volumes/cat/school_gold/gold_volume/model_cards/run/model-card-retention.pdf",
            "20260302",
        )
        == "model-card-retention_20260302.pdf"
    )


def test_resolve_latest_model_card_pdf_includes_run_date(tmp_path, monkeypatch):
    run_dir = tmp_path / "e39e2308"
    run_dir.mkdir()
    (run_dir / "model-card-retention.pdf").write_text("%PDF", encoding="utf-8")
    start_ms = int(
        datetime(2026, 3, 2, 15, 49, tzinfo=timezone.utc).timestamp() * 1000
    )
    monkeypatch.setattr(
        mr,
        "gold_model_cards_run_dir",
        lambda _catalog, _institution_id, _run_id: str(run_dir),
    )
    client = FakeMlflowClient(
        models=[_registered("staging.school_gold.retention", 1)],
        versions_by_name={
            "staging.school_gold.retention": [_version(1, "e39e2308")],
        },
        runs_by_id={
            "e39e2308": SimpleNamespace(info=SimpleNamespace(start_time=start_ms)),
        },
    )
    pdf_path, run_id, full_name, version, yyyymmdd = mr.resolve_latest_model_card_pdf(
        client,
        catalog="staging",
        institution_id="school",
    )
    assert run_id == "e39e2308"
    assert full_name == "staging.school_gold.retention"
    assert version == "1"
    assert yyyymmdd == "20260302"
    assert pdf_path.endswith("model-card-retention.pdf")
