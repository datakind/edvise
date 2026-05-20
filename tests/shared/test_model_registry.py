"""Tests for edvise.shared.model_registry."""

from __future__ import annotations

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


class FakeMlflowClient:
    def __init__(
        self,
        models: list[SimpleNamespace] | None = None,
        versions_by_name: dict[str, list[SimpleNamespace]] | None = None,
    ):
        self._models = models or []
        self._versions_by_name = versions_by_name or {}

    def search_registered_models(self):
        return self._models

    def search_model_versions(self, filter_string: str):
        # filter_string is name='catalog.schema.model'
        name = filter_string.split("'", 2)[1]
        return self._versions_by_name.get(name, [])


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


def test_gold_model_cards_run_dir():
    assert mr.gold_model_cards_run_dir("staging_sst_01", "jf_drake_state_cc", "abc") == (
        "/Volumes/staging_sst_01/jf_drake_state_cc_gold/gold_volume/model_cards/abc"
    )
