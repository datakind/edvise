import argparse

from edvise.configs import es as es_configs
from edvise.configs import pdp as pdp_configs
from edvise.reporting.model_card.h2o_es import H2OESModelCard
from edvise.reporting.model_card.h2o_pdp import H2OPDPModelCard
from edvise.scripts.training_h2o import resolve_spec


def test_resolve_spec_pdp_uses_pdp_model_card():
    args = argparse.Namespace(schema_type="pdp", features_table_path=None)
    spec = resolve_spec(args)

    assert spec.schema_type == "pdp"
    assert spec.cfg_schema is pdp_configs.PDPProjectConfig
    assert spec.model_card_cls is H2OPDPModelCard


def test_resolve_spec_edvise_uses_es_model_card():
    args = argparse.Namespace(schema_type="edvise", features_table_path=None)
    spec = resolve_spec(args)

    assert spec.schema_type == "edvise"
    assert spec.cfg_schema is es_configs.ESProjectConfig
    assert spec.model_card_cls is H2OESModelCard
