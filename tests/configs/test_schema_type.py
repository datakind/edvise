import pytest

from edvise.configs.es import ESProjectConfig
from edvise.configs.legacy import LegacyProjectConfig
from edvise.configs.pdp import PDPProjectConfig
from edvise.configs.schema_type import (
    is_legacy_schema,
    project_config_class,
    resolve_features_table_path,
)


@pytest.mark.parametrize(
    ("schema_type", "expected"),
    [
        ("pdp", PDPProjectConfig),
        ("edvise", ESProjectConfig),
        ("es", ESProjectConfig),
        ("legacy", LegacyProjectConfig),
    ],
)
def test_project_config_class(schema_type, expected):
    assert project_config_class(schema_type) is expected


def test_project_config_class_unknown_raises():
    with pytest.raises(ValueError, match="Unknown --schema_type"):
        project_config_class("unknown")


def test_is_legacy_schema():
    assert is_legacy_schema("legacy")
    assert not is_legacy_schema("pdp")


def test_resolve_features_table_path_default_for_pdp():
    assert (
        resolve_features_table_path("pdp", None)
        == "shared/assets/pdp_features_table.toml"
    )


def test_resolve_features_table_path_override():
    assert resolve_features_table_path("pdp", "custom/features.toml") == (
        "custom/features.toml"
    )


def test_resolve_features_table_path_legacy_requires_override():
    with pytest.raises(ValueError, match="--features_table_path required"):
        resolve_features_table_path("legacy", None)
