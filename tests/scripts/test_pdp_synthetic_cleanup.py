from edvise.scripts.pdp_synthetic_cleanup import is_institution_gold_model


def test_is_institution_gold_model_matches_pdp_model_names():
    catalog = "dev_sst_02"
    institution = "synthetic_integration"
    assert is_institution_gold_model(
        "dev_sst_02.synthetic_integration_gold.retention_into_year_2_bachelors",
        catalog,
        institution,
    )


def test_is_institution_gold_model_matches_legacy_h2o_model_names():
    catalog = "dev_sst_02"
    institution = "synthetic_integration"
    assert is_institution_gold_model(
        "dev_sst_02.synthetic_integration_gold.synthetic_integration_retention_2_year_time_first_within_cohort_h2o_automl",
        catalog,
        institution,
    )


def test_is_institution_gold_model_rejects_other_schemas():
    catalog = "dev_sst_02"
    institution = "synthetic_integration"
    assert not is_institution_gold_model(
        "dev_sst_02.other_institution_gold.retention_into_year_2_bachelors",
        catalog,
        institution,
    )
