import pytest
import pydantic as pyd

from edvise.configs.es import FeaturesConfig as ESFeaturesConfig
from edvise.configs.pdp import FeaturesConfig as PDPFeaturesConfig


@pytest.mark.parametrize("config_cls", [PDPFeaturesConfig, ESFeaturesConfig])
def test_first_term_of_year_defaults_to_fall(config_cls):
    assert config_cls().first_term_of_year == "FALL"
    assert config_cls(first_term_of_year="SUMMER").first_term_of_year == "SUMMER"


@pytest.mark.parametrize("config_cls", [PDPFeaturesConfig, ESFeaturesConfig])
@pytest.mark.parametrize("raw", ["summer", "Summer", " SUMMER "])
def test_first_term_of_year_ignores_case(config_cls, raw):
    assert config_cls(first_term_of_year=raw).first_term_of_year == "SUMMER"


@pytest.mark.parametrize("config_cls", [PDPFeaturesConfig, ESFeaturesConfig])
def test_first_term_of_year_rejects_other_seasons(config_cls):
    with pytest.raises(pyd.ValidationError):
        config_cls(first_term_of_year="WINTER")
