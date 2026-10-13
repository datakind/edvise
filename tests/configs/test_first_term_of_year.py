import pytest
import pydantic as pyd

from edvise.configs.es import FeaturesConfig as ESFeaturesConfig
from edvise.configs.pdp import FeaturesConfig as PDPFeaturesConfig


def test_es_first_term_of_year_defaults_to_fall():
    assert ESFeaturesConfig().first_term_of_year == "FALL"
    assert ESFeaturesConfig(first_term_of_year="SUMMER").first_term_of_year == "SUMMER"


def test_es_first_term_of_year_rejects_other_seasons():
    with pytest.raises(pyd.ValidationError):
        ESFeaturesConfig(first_term_of_year="WINTER")


def test_pdp_features_config_does_not_accept_first_term_of_year():
    assert "first_term_of_year" not in PDPFeaturesConfig.model_fields
