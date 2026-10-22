import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from edvise.scripts.legacy_postprocessing import (
    legacy_postprocess_script_exists,
    should_defer_model_card_to_postprocessing,
    write_model_card_after_bias_mitigation,
)


def _cfg(*, group="18-24", threshold=0.3, enabled=True, mitigation=True):
    bias = None
    if mitigation:
        bias = SimpleNamespace(student_group=group, custom_threshold=threshold)
    return SimpleNamespace(
        modeling=SimpleNamespace(bias_mitigation=bias),
        postprocessing=SimpleNamespace(enabled=enabled),
        preprocessing=object(),
        institution_id="southeast_cc",
    )


def test_defer_only_for_legacy_bias_mitigation_with_postprocessing():
    cfg = _cfg()
    assert should_defer_model_card_to_postprocessing(
        schema_type="legacy",
        cfg=cfg,
        postprocessing_enabled=True,
        postprocess_script_exists=True,
    )


@pytest.mark.parametrize(
    ("schema_type", "enabled", "script_exists", "mitigation"),
    [
        ("pdp", True, True, True),
        ("edvise", True, True, True),
        ("legacy", False, True, True),
        ("legacy", True, False, True),
        ("legacy", True, True, False),
    ],
)
def test_do_not_defer_model_card(schema_type, enabled, script_exists, mitigation):
    cfg = _cfg(enabled=enabled, mitigation=mitigation)
    assert not should_defer_model_card_to_postprocessing(
        schema_type=schema_type,
        cfg=cfg,
        postprocessing_enabled=enabled,
        postprocess_script_exists=script_exists,
    )


def test_postprocess_script_exists_for_school_folder(tmp_path):
    school = tmp_path / "southeast_cc" / "fall_to_spring"
    school.mkdir(parents=True)
    (school / "postprocessing.py").write_text("def run(**kwargs):\n    return None\n")
    args = SimpleNamespace(
        databricks_institution_name="southeast_cc",
        model_name="dev_sst_02.southeast_cc_gold.fall_to_spring",
        DB_workspace="dev_sst_02",
        ds_run_as="",
        ssi_pipelines_workspace_root=str(tmp_path),
    )
    assert legacy_postprocess_script_exists(args)


def test_missing_postprocess_script_keeps_training_card(tmp_path):
    args = SimpleNamespace(
        databricks_institution_name="southeast_cc",
        model_name="fall_to_spring",
        DB_workspace="dev_sst_02",
        ds_run_as="",
        ssi_pipelines_workspace_root=str(tmp_path),
    )
    assert not legacy_postprocess_script_exists(args)


def _stub_card_imports(card_cls, model_name):
    """Avoid importing the H2O stack while the writer resolves its card class."""
    modeling = ModuleType("edvise.modeling")
    modeling.__path__ = []
    registration = ModuleType("edvise.modeling.registration")
    registration.get_model_name_from_config = lambda **_kwargs: model_name
    model_card_pkg = ModuleType("edvise.reporting.model_card")
    model_card_pkg.__path__ = []
    legacy_card = ModuleType("edvise.reporting.model_card.h2o_legacy")
    legacy_card.H2OLegacyModelCard = card_cls
    saved = {}
    names = (
        "edvise.modeling",
        "edvise.modeling.registration",
        "edvise.reporting.model_card",
        "edvise.reporting.model_card.h2o_legacy",
    )
    for name in names:
        saved[name] = sys.modules.get(name)
    sys.modules["edvise.modeling"] = modeling
    sys.modules["edvise.modeling.registration"] = registration
    sys.modules["edvise.reporting.model_card"] = model_card_pkg
    sys.modules["edvise.reporting.model_card.h2o_legacy"] = legacy_card
    return saved


def _restore_modules(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_write_model_card_after_bias_mitigation(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(
        "edvise.dataio.read.read_config",
        lambda *_args, **_kwargs: cfg,
    )
    card = MagicMock()
    card_cls = MagicMock(return_value=card)
    saved = _stub_card_imports(card_cls, "retention_fall_to_spring")
    try:
        write_model_card_after_bias_mitigation(
            "config.toml",
            catalog="dev_sst_02",
        )
    finally:
        _restore_modules(saved)

    card_cls.assert_called_once_with(
        config=cfg,
        catalog="dev_sst_02",
        model_name="retention_fall_to_spring",
    )
    card.build.assert_called_once_with()
    card.reload_card.assert_called_once_with()
    card.export_to_pdf.assert_called_once_with()


def test_write_model_card_skips_when_bias_mitigation_unset(monkeypatch):
    cfg = _cfg(mitigation=False)
    monkeypatch.setattr(
        "edvise.dataio.read.read_config",
        lambda *_args, **_kwargs: cfg,
    )
    card_cls = MagicMock()
    saved = _stub_card_imports(card_cls, "unused")
    try:
        write_model_card_after_bias_mitigation("config.toml", catalog="dev_sst_02")
    finally:
        _restore_modules(saved)

    card_cls.assert_not_called()
