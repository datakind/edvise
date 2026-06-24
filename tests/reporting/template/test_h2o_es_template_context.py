import pytest
import pandas as pd
import re
from unittest.mock import patch

from edvise.configs.es import ESProjectConfig
from edvise.reporting.model_card.h2o_es import H2OESModelCard


def make_es_config() -> ESProjectConfig:
    return ESProjectConfig(
        institution_id="inst_id",
        institution_name="Inst Name",
        model={"experiment_id": "exp123", "run_id": "abc"},
        datasets={
            "raw_course": "dummy.csv",
            "raw_cohort": "dummy.csv",
        },
        preprocessing={
            "selection": {"student_criteria": {"status": "active"}},
            "checkpoint": {
                "name": "checkpoint_nth",
                "type_": "nth",
                "n": 4,
            },
            "target": {
                "name": "retention",
                "type_": "retention",
                "max_academic_year": "2025-26",
            },
            "features": {},
        },
        modeling={
            "feature_selection": {
                "collinear_threshold": 10.0,
                "low_variance_threshold": 0.0,
                "incomplete_threshold": 0.5,
            },
            "training": {"primary_metric": "logloss", "timeout_minutes": 10},
        },
        split_col=None,
    )


@pytest.mark.parametrize("card_class", [H2OESModelCard])
@patch("edvise.reporting.sections.registry.SectionRegistry.render_all")
@patch("edvise.reporting.model_card.h2o_es.H2OESModelCard.collect_metadata")
@patch("edvise.reporting.model_card.h2o_es.H2OESModelCard.load_model")
@patch("edvise.reporting.model_card.h2o_es.H2OESModelCard.extract_training_data")
@patch("edvise.reporting.model_card.h2o_es.H2OESModelCard.find_model_version")
def test_template_placeholders_are_in_context(
    mock_find_version,
    mock_extract_data,
    mock_load_model,
    mock_collect_metadata,
    mock_render_all,
    card_class,
):
    config = make_es_config()

    card = card_class(config=config, catalog="demo", model_name="test_model")

    mock_load_model.side_effect = lambda: (
        setattr(card, "run_id", "dummy_run_id")
        or setattr(card, "experiment_id", "dummy_experiment_id")
        or setattr(card, "model", object())
        or setattr(card, "training_data", pd.DataFrame(columns=["sample_weight"]))
        or setattr(card, "modeling_data", pd.DataFrame({"learner_id": []}))
    )

    mock_collect_metadata.side_effect = lambda: card.context.update(
        {
            "model_version": "12",
            "artifact_path": "dummy/path",
            "training_dataset_size": 100,
            "number_of_features": 20,
            "feature_importances_by_shap_plot": "![shap](shap.png)",
            "test_confusion_matrix": "confusion_matrix.png",
            "test_roc_curve": "roc_curve.png",
            "test_calibration_curve": "calibration_curve.png",
            "test_histogram": "histogram.png",
            "model_comparison_plot": "comparison.png",
            "collinearity_threshold": 10.0,
            "low_variance_threshold": 0.0,
            "incomplete_threshold": 0.5,
            "funnel_image": "data_funnel.png",
        }
    )

    mock_render_all.return_value = {
        "primary_metric_section": "Primary metric content",
        "checkpoint_section": "Checkpoint content",
        "bias_summary_section": "Bias summary",
        "performance_by_splits_section": "Performance content",
        "evaluation_by_group_section": "Group evaluation",
        "logo": "logo.png",
        "target_population_section": "Population info",
        "institution_name": "Test University",
        "sample_weight_section": "Sample weight info",
        "data_split_table": "Data split table",
        "classification_threshold_section": "Classification threshold: 0.5",
        "bias_groups_section": "Bias groups",
        "selected_features_ranked_by_shap": "Feature list",
        "development_note_section": "Dev note",
        "outcome_section": "Outcome explanation",
    }

    card.load_model()
    card.find_model_version()
    card.extract_training_data()
    card._register_sections()
    card.collect_metadata()

    card.context.update(card.section_registry.render_all())

    with open(card.template_path, "r") as f:
        template = f.read()

    matches = set(re.findall(r"{([\w_]+)}", template))
    missing = matches - card.context.keys()
    assert not missing, f"Missing context keys for template: {missing}"
