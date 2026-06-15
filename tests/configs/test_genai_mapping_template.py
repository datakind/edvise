"""Validate GenAI mapping config templates against pydantic schemas."""

try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

import pathlib

import pydantic as pyd

from edvise.configs.genai import IdentityAgentInputsConfig

SRC_ROOT = pathlib.Path(__file__).parents[2] / "configs" / "genai_mapping"


def test_template_identity_agent_inputs_toml() -> None:
    config_path = SRC_ROOT / "inputs-TEMPLATE.toml"
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    result = IdentityAgentInputsConfig.model_validate(cfg)
    assert isinstance(result, pyd.BaseModel)
    assert result.institution_id == "inst_id"
    assert "student" in result.datasets.onboard_files
