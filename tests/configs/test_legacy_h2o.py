try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pydantic as pyd
import pathlib
import pytest

from edvise.configs import legacy

SRC_ROOT = pathlib.Path(__file__).parents[2] / "configs" / "custom_h2o"


@pytest.fixture(scope="module")
def template_cfg_dict():
    config_path = SRC_ROOT / "config-TEMPLATE.toml"
    with config_path.open("rb") as f:
        return tomllib.load(f)


def test_template_legacy_cfgs(template_cfg_dict):
    result = legacy.LegacyProjectConfig.model_validate(template_cfg_dict)
    print(result)
    assert isinstance(result, pyd.BaseModel)


@pytest.mark.parametrize(
    ["cfg_str", "context"],
    [
        (
            'institution_id = "custom_inst_id"',
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "INVALID_IDENTIFIER!"
            institution_name = "Custom Institution Name"
            """,
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "custom_inst_id"
            institution_name = "Custom Institution Name"
            [datasets.bronze]

            [datasets.bronze.raw_cohort]
            primary_keys = ["student_id"]
            non_null_cols = ["acad_year"]
            train_file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_train.csv"
            predict_file_path = "/Volumes/CATALOG/INST_ID_bronze/.../FILE_NAME_cohort_inference.csv"
            """,
            pytest.raises(pyd.ValidationError),
        ),
        (
            """
            institution_id = "inst_id"
            institution_name = "Inst Name"

            [model]
            experiment_id = "EXPERIMENT_ID"
            run_id = "RUN_ID"
            """,
            pytest.raises(pyd.ValidationError),
        ),
    ],
)
def test_bad_legacy_cfgs(cfg_str, context):
    cfg = tomllib.loads(cfg_str)
    with context:
        result = legacy.LegacyProjectConfig.model_validate(cfg)
        assert isinstance(result, pyd.BaseModel)


def test_substitute_uc_catalog_in_string():
    assert legacy.substitute_uc_catalog_in_string(
        "CATALOG.foo.bar", "dev_sst_02"
    ) == "dev_sst_02.foo.bar"
    assert legacy.substitute_uc_catalog_in_string(
        "/Volumes/CATALOG/inst_bronze/x.csv", "dev_sst_02"
    ) == "/Volumes/dev_sst_02/inst_bronze/x.csv"
    assert legacy.substitute_uc_catalog_in_string(
        "dbfs:/Volumes/CATALOG/inst_bronze/x.csv", "dev_sst_02"
    ) == "dbfs:/Volumes/dev_sst_02/inst_bronze/x.csv"
    assert legacy.substitute_uc_catalog_in_string("unchanged", "x") == "unchanged"


def test_apply_runtime_uc_catalog_on_template():
    template_path = (
        pathlib.Path(__file__).parents[2] / "configs" / "legacy_h2o" / "config-TEMPLATE.toml"
    )
    with template_path.open("rb") as f:
        raw = tomllib.load(f)
    cfg = legacy.LegacyProjectConfig.model_validate(raw)
    resolved = legacy.apply_runtime_uc_catalog(cfg, "dev_sst_02")
    student = resolved.datasets.bronze["raw_student"]
    assert student.train_file_path
    assert "/Volumes/dev_sst_02/" in student.train_file_path
    assert "CATALOG" not in student.train_file_path
    mt = resolved.datasets.silver.modeling.train_table_path
    assert mt is not None
    assert mt.startswith("dev_sst_02.")
    assert "CATALOG" not in mt
