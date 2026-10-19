from types import SimpleNamespace

import pandas as pd
import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from edvise.reporting.sections.registry import SectionRegistry
from edvise.reporting.sections.legacy import (
    bias_sections as legacy_bias_sections,
)
from edvise.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card():
    card = MagicMock()
    formatter = Formatting()
    card.format.indent_level.side_effect = formatter.indent_level
    card.format.friendly_case.side_effect = formatter.friendly_case
    card.format.header_level.side_effect = formatter.header_level
    card.format.bold.side_effect = formatter.bold
    card.format.italic.side_effect = formatter.italic
    card.format.format_threshold.side_effect = formatter.format_threshold
    card.format.format_rate.side_effect = formatter.format_rate
    card.assets_folder = "/tmp/assets"
    card.run_id = "dummy_run_id"
    card.cfg.modeling.bias_mitigation = None
    return card


@pytest.fixture
def registry():
    return SectionRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: bias_groups_section
# ─────────────────────────────────────────────────────────────────────────────


@patch(
    "edvise.reporting.sections.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=["firstgenflag", "gender", "ethnicity_ipeds", "demo_race"],
)
def test_bias_groups_section_with_valid_aliases(mock_groups, mock_card, registry):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "First-Generation Status",
        "gender": "Gender",
        "ethnicity_ipeds": "Ethnicity",
        "demo_race": "Race",
    }

    legacy_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]

    assert (
        "- Our assessment for FNR Parity was conducted across the following student groups."
        in result
    )
    assert "- First-Generation Status" in result
    assert "- Gender" in result
    assert "- Ethnicity" in result
    assert "- Race" in result
    assert "one threshold" not in result
    assert "Support Label" not in result
    mock_groups.assert_called_with(mock_card.run_id)


@patch(
    "edvise.reporting.sections.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=["firstgenflag", "disabilityflag"],
)
def test_bias_groups_section_with_aliases_that_need_friendlycase(
    mock_groups, mock_card
):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "first_generation_status",
        "disabilityflag": "disability_status",
    }

    registry = SectionRegistry()
    legacy_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]

    assert "- First Generation Status" in result
    assert "- Disability Status" in result


@patch(
    "edvise.reporting.sections.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=[],
)
def test_bias_groups_section_with_missing_artifacts(mock_groups, mock_card, caplog):
    mock_card.cfg.student_group_aliases = {
        "gender": "Gender",
    }

    registry = SectionRegistry()
    legacy_bias_sections.register_bias_sections(mock_card, registry)

    with caplog.at_level("WARNING"):
        rendered = registry.render_all()

    result = rendered["bias_groups_section"]
    assert "- Unable to determine assessed student groups" in result
    assert any(
        "No bias_test group_metrics artifacts found" in message
        for message in caplog.messages
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: bias_summary_section uses aliases in header
# ─────────────────────────────────────────────────────────────────────────────


@patch("edvise.reporting.utils.utils.download_artifact")
def test_bias_summary_section_uses_aliases(mock_download_artifact, mock_card, tmp_path):
    import pandas as pd

    # Setup alias
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "First-Generation",
    }

    mock_card.format.friendly_case.side_effect = lambda x: x.replace("_", " ").title()
    mock_card.assets_folder = tmp_path
    mock_card.run_id = "run123"

    # Write test CSV
    df = pd.DataFrame(
        {
            "group": ["firstgenflag"],
            "split_name": ["test"],
            "subgroups": ["yes vs no"],
            "fnr_percentage_difference": [0.12],
            "type": ["p < 0.05, 95% CI [0.05, 0.19]"],
        }
    )
    bias_path = tmp_path / "bias_flags"
    bias_path.mkdir()
    df.to_csv(bias_path / "high_bias_flags.csv", index=False)

    # Patch return of download_artifact
    def download_side_effect(run_id, local_folder, artifact_path, description=None):
        return str(tmp_path / artifact_path)  # ← ensure string return

    mock_download_artifact.side_effect = download_side_effect

    # Run
    registry = SectionRegistry()
    legacy_bias_sections.register_bias_sections(mock_card, registry)
    rendered = registry.render_all()
    result = rendered["bias_summary_section"]

    assert "First-Generation" in result
    assert "12% difference" in result
    assert "####Disparities by Student Group" in result
    assert "Fairness Adjustment" not in result
    assert "Support Label" not in result
    assert "are not available for this model" not in result


def _attach_bias_flags(mock_card, tmp_path, group):
    df = pd.DataFrame(
        {
            "group": [group],
            "split_name": ["test"],
            "subgroups": ["yes vs no"],
            "fnr_percentage_difference": [0.12],
            "type": ["p < 0.05, 95% CI [0.05, 0.19]"],
        }
    )
    bias_path = tmp_path / "bias_flags"
    bias_path.mkdir()
    df.to_csv(bias_path / "high_bias_flags.csv", index=False)
    mock_card.assets_folder = tmp_path
    mock_card.run_id = "run123"

    def download_side_effect(run_id, local_folder, artifact_path, **_kwargs):
        return str(tmp_path / artifact_path)

    return download_side_effect


def _render_legacy_bias(mock_card):
    registry = SectionRegistry()
    legacy_bias_sections.register_bias_sections(mock_card, registry)
    return registry.render_all()


@patch(
    "edvise.reporting.utils.utils.list_paths_in_directory",
    return_value=[],
)
@patch("edvise.reporting.utils.utils.download_artifact")
def test_bias_summary_with_mitigation_includes_logged_fnr(
    mock_download_artifact, _mock_list_paths, mock_card, tmp_path
):
    mock_card.cfg.student_group_aliases = {"agegroup": "Age Group"}
    mock_card.cfg.modeling.bias_mitigation = SimpleNamespace(
        student_group_col="agegroup",
        student_group_col_alias="Age",
        student_group="25 and over",
        custom_threshold=0.35,
    )
    mock_card.cfg.inference.min_prob_pos_label = 0.5
    run = MagicMock()
    run.data.params = {"bias_mitigation_custom_threshold": "0.35"}
    run.data.metrics = {
        "bias_mitigation_group_fnr": 0.12,
        "bias_mitigation_reference_fnr": 0.2,
        "bias_mitigation_fnr_abs_gap": 0.08,
    }
    mock_card.client.get_run.return_value = run
    mock_download_artifact.side_effect = _attach_bias_flags(
        mock_card, tmp_path, "agegroup"
    )

    rendered = _render_legacy_bias(mock_card)
    summary = rendered["bias_summary_section"]
    groups = rendered["bias_groups_section"]

    assert "####Disparities Before the Fairness Adjustment" in summary
    assert "These results use one threshold (0.5) for all students." in summary
    assert "####Disparities by Student Group" not in summary
    assert "Age Group" in summary
    assert "12% difference" not in summary
    assert "No statistically significant disparities were found" in summary
    assert "####Fairness Adjustment" in summary
    assert "across the subgroups of Age Group" in summary
    assert "`agegroup`" not in summary
    assert "- Threshold for" not in summary
    assert "Every student gets the same Support Score." in summary
    assert "for **25 and over** students and **0.5** for all other students" in summary
    assert "underpredicts for **12.0%** of 25 and over students" in summary
    assert "compared with **20.0%** of all other students" in summary
    assert "a gap of **8.0 percentage points**" in summary
    assert (
        "_The comparison before the adjustment, at one threshold of 0.5, "
        "is not in the threshold search results._" in summary
    )
    assert "Before the adjustment, with one threshold of **0.5**" not in summary
    assert "**Tradeoff.**" in summary
    assert "are not available for this model" not in summary
    assert (
        "In the advisor output file, the Support Label appears alongside the "
        "Support Score and reflects this fairness adjustment." in summary
    )
    assert "These results use one threshold (0.5) for all students." in groups
    mock_card.client.get_run.assert_called_once_with("run123")


@patch(
    "edvise.reporting.utils.utils.list_paths_in_directory",
    return_value=[],
)
@patch("edvise.reporting.utils.utils.download_artifact")
def test_bias_summary_with_mitigation_config_only_when_metrics_missing(
    mock_download_artifact, _mock_list_paths, mock_card, tmp_path
):
    mock_card.cfg.student_group_aliases = {"gender": "Gender"}
    mock_card.cfg.modeling.bias_mitigation = SimpleNamespace(
        student_group_col="pell_flag",
        student_group_col_alias="Pell Status",
        student_group="recipient",
        custom_threshold=0.4,
    )
    mock_card.cfg.inference = None
    run = MagicMock()
    run.data.params = {}
    run.data.metrics = {}
    mock_card.client.get_run.return_value = run
    mock_download_artifact.side_effect = _attach_bias_flags(
        mock_card, tmp_path, "gender"
    )

    summary = _render_legacy_bias(mock_card)["bias_summary_section"]

    assert "####Disparities Before the Fairness Adjustment" in summary
    assert "#####Gender" in summary
    assert "12% difference" in summary
    assert "####Fairness Adjustment" in summary
    assert "across the subgroups of Pell Status" in summary
    assert "`pell_flag`" not in summary
    assert "**recipient**" in summary
    assert "**0.4**" in summary
    assert "**0.5**" in summary
    assert "Results after the fairness adjustment are not available" in summary
    assert "Effect on results" not in summary
    assert "threshold search results" not in summary
    assert (
        "In the advisor output file, the Support Label appears alongside the "
        "Support Score and reflects this fairness adjustment." in summary
    )


@patch(
    "edvise.reporting.utils.utils.list_paths_in_directory",
    return_value=[],
)
@patch("edvise.reporting.utils.utils.download_artifact")
def test_bias_summary_documents_rule_when_mlflow_read_fails(
    mock_download_artifact, _mock_list_paths, mock_card, tmp_path
):
    mock_card.cfg.student_group_aliases = {"student_group": "Student Group"}
    mock_card.cfg.modeling.bias_mitigation = SimpleNamespace(
        student_group_col="student_group",
        student_group="freshmen",
        custom_threshold=0.45,
    )
    mock_card.cfg.inference.min_prob_pos_label = None
    mock_card.client.get_run.side_effect = RuntimeError("mlflow unavailable")
    mock_download_artifact.side_effect = _attach_bias_flags(
        mock_card, tmp_path, "student_group"
    )

    summary = _render_legacy_bias(mock_card)["bias_summary_section"]

    assert "**freshmen**" in summary
    assert "**0.45**" in summary
    assert "**0.5**" in summary
    assert "across the subgroups of Student Group" in summary
    assert "Results after the fairness adjustment are not available" in summary
    assert "12% difference" not in summary


@patch("edvise.reporting.utils.utils.list_paths_in_directory")
@patch("edvise.reporting.utils.utils.download_artifact")
def test_bias_summary_reads_fnr_from_threshold_artifact(
    mock_download_artifact, mock_list_paths, mock_card, tmp_path
):
    mock_card.cfg.student_group_aliases = {"age_bias_group": "Age Bias Group"}
    mock_card.cfg.modeling.bias_mitigation = SimpleNamespace(
        student_group_col="age_bias_group",
        student_group_col_alias="Age",
        student_group="18-24",
        custom_threshold=0.3,
    )
    mock_card.cfg.inference.min_prob_pos_label = 0.5
    run = MagicMock()
    run.data.params = {}
    run.data.metrics = {}
    mock_card.client.get_run.return_value = run

    artifact_name = "bias_mitigation/bias_mitigation_thresholds_oc7qn6v_.csv"
    frame = pd.DataFrame(
        {
            "group": ["18-24", "18-24", "18-24", "18-24"],
            "threshold": [0.2, 0.3, 0.4, 0.5],
            "reference_fnr": [0.21621621621621623] * 4,
            "group_fnr": [0.05, 0.2074074074074074, 0.29074074074074074, 0.42],
            "fnr_abs_gap": [
                0.16621621621621624,
                0.008808808808808827,
                0.07452452452452452,
                0.20378378378378377,
            ],
            "within_tolerance": [False, True, False, False],
            "balanced_accuracy": [0.64, 0.69, 0.74, 0.60],
        }
    )
    artifact_path = tmp_path / artifact_name
    artifact_path.parent.mkdir(parents=True)
    frame.to_csv(artifact_path, index=False)
    mock_list_paths.return_value = [artifact_name]
    mock_download_artifact.side_effect = _attach_bias_flags(
        mock_card, tmp_path, "age_bias_group"
    )

    summary = _render_legacy_bias(mock_card)["bias_summary_section"]

    assert "**18-24**" in summary
    assert "**0.3**" in summary
    assert "**20.7%**" in summary
    assert "**21.6%**" in summary
    assert "**0.9 percentage points**" in summary
    assert "Before the adjustment, with one threshold of **0.5**" in summary
    assert "underpredicts for **42.0%** of 18-24 students" in summary
    assert "a gap of **20.4 percentage points**" in summary
    assert summary.index("Before the adjustment") < summary.index(
        "After the adjustment"
    )
    assert "is not in the threshold search results" not in summary
    assert "are not available for this model" not in summary
    mock_list_paths.assert_any_call("run123", "bias_mitigation")


def test_default_threshold_row_does_not_fall_back_to_the_chosen_threshold():
    frame = pd.DataFrame(
        {
            "threshold": [0.3],
            "group_fnr": [0.2074074074074074],
            "reference_fnr": [0.21621621621621623],
            "fnr_abs_gap": [0.008808808808808827],
        }
    )

    assert legacy_bias_sections._matching_threshold_row(frame, 0.5) is None
    chosen = legacy_bias_sections._select_threshold_row(frame, 0.3)
    assert float(chosen["threshold"]) == 0.3


@patch(
    "edvise.reporting.utils.utils.list_paths_in_directory",
    return_value=[],
)
@patch("edvise.reporting.utils.utils.download_artifact", return_value=None)
def test_group_column_omits_generic_alias_for_custom_column(
    _mock_download_artifact, _mock_list_paths, mock_card
):
    mock_card.cfg.student_group_aliases = {}
    mock_card.cfg.modeling.bias_mitigation = SimpleNamespace(
        student_group_col="agegroup",
        student_group_col_alias="Student Group",
        student_group="25 and over",
        custom_threshold=0.3,
    )
    mock_card.cfg.inference.min_prob_pos_label = 0.55
    run = MagicMock()
    run.data.params = {}
    run.data.metrics = {}
    mock_card.client.get_run.return_value = run

    summary = _render_legacy_bias(mock_card)["bias_summary_section"]

    assert "across the subgroups of Agegroup" in summary
    assert "**Student Group**" not in summary
    assert "**0.55**" in summary
    assert "**0.3**" in summary
    assert "Results after the fairness adjustment are not available" in summary
    assert "Disparities After the Fairness Adjustment" not in summary


def test_bias_summary_shows_post_mitigation_group_plots(mock_card, tmp_path):
    mock_card.assets_folder = str(tmp_path)
    mock_card.cfg.student_group_cols = ["gender", "ethnicity", "age_group"]
    mock_card.cfg.student_group_aliases = {
        "age_bias_group": "Age Bias Group",
        "gender": "Gender",
    }
    mock_card.cfg.modeling.bias_mitigation = SimpleNamespace(
        student_group_col="age_bias_group",
        student_group_col_alias="Age",
        student_group="18-24",
        custom_threshold=0.3,
    )
    mock_card.cfg.inference.min_prob_pos_label = 0.5
    run = MagicMock()
    run.data.params = {}
    run.data.metrics = {}
    mock_card.client.get_run.return_value = run

    pre_flags = tmp_path / "bias_flags" / "high_bias_flags.csv"
    pre_flags.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "group": "age_group",
                "split_name": "test",
                "subgroups": "20-21 vs 50-64",
                "fnr_percentage_difference": 0.23,
                "type": "a significant difference",
            },
            {
                "group": "age_bias_group",
                "split_name": "test",
                "subgroups": "18-24 vs other",
                "fnr_percentage_difference": 0.2,
                "type": "a significant difference",
            },
        ]
    ).to_csv(pre_flags, index=False)
    post_flags = (
        tmp_path
        / "bias_flags"
        / "post_mitigation"
        / "post_mitigation_low_bias_flags.csv"
    )
    post_flags.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "group": "gender",
                "split_name": "test",
                "subgroups": "Female vs Male",
                "fnr_percentage_difference": 0.12,
                "type": "a significant difference",
            },
            {
                "group": "age_bias_group",
                "split_name": "test",
                "subgroups": "other vs 18-24",
                "fnr_percentage_difference": 0.06,
                "type": "overlapping confidence intervals with a p-value of 0.027",
            },
        ]
    ).to_csv(post_flags, index=False)
    age_plot = (
        tmp_path / "fnr_plots" / "post_mitigation" / "test_age_bias_group_fnr.png"
    )
    age_plot.parent.mkdir(parents=True)
    age_plot.write_bytes(b"png")
    gender_plot = tmp_path / "fnr_plots" / "post_mitigation" / "test_gender_fnr.png"
    gender_plot.write_bytes(b"png")
    rates = tmp_path / "fnr_plots" / "post_mitigation" / "fnr_rates.csv"
    pd.DataFrame(
        [
            {
                "group": "age_bias_group",
                "subgroup": "other",
                "split_name": "test",
                "fnr": 0.21621621621621623,
            },
            {
                "group": "age_bias_group",
                "subgroup": "18-24",
                "split_name": "test",
                "fnr": 0.2074074074074074,
            },
            {
                "group": "gender",
                "subgroup": "Male",
                "split_name": "test",
                "fnr": 0.23,
            },
            {
                "group": "gender",
                "subgroup": "Female",
                "split_name": "train",
                "fnr": 0.99,
            },
            {
                "group": "gender",
                "subgroup": "Female",
                "split_name": "test",
                "fnr": 0.195,
            },
            {
                "group": "age_group",
                "subgroup": "65-99",
                "split_name": "test",
                "fnr": 0.098,
            },
            {
                "group": "age_group",
                "subgroup": "Other",
                "split_name": "test",
                "fnr": 0.0,
            },
        ]
    ).to_csv(rates, index=False)

    def _download(**kwargs):
        return str(tmp_path / kwargs["artifact_path"])

    with (
        patch("edvise.reporting.utils.utils.download_artifact", side_effect=_download),
        patch(
            "edvise.reporting.utils.utils.list_paths_in_directory",
            return_value=[],
        ),
    ):
        summary = _render_legacy_bias(mock_card)["bias_summary_section"]

    assert "Disparities Before the Fairness Adjustment" in summary
    assert "#####Age" in summary
    assert "23% difference" in summary
    assert "20% difference" not in summary
    assert "Disparities After the Fairness Adjustment" in summary
    assert "for **18-24** students and **0.5** for 25-99 students" in summary
    assert (
        "These results use the group-specific threshold described under "
        "Fairness Adjustment." in summary
    )
    assert summary.index("####Fairness Adjustment") < summary.index(
        "Disparities Before the Fairness Adjustment"
    )
    assert summary.index("Disparities Before the Fairness Adjustment") < summary.index(
        "Disparities After the Fairness Adjustment"
    )
    assert "#####Age Bias Group" not in summary
    assert "**_25-99_**" not in summary
    assert "other" not in summary.lower()
    assert "**_Other_**" not in summary
    assert "FNR:" not in summary
    assert str(age_plot) not in summary
    assert "#####Gender" in summary
    assert "6% difference" not in summary
    assert "12% difference" in summary
    assert str(gender_plot) in summary
    assert "ethnicity" not in summary.lower()
    assert "Results after the fairness adjustment are not available" in summary


def test_reference_age_display_label_uses_latest_bin_and_keeps_fine_other():
    label = legacy_bias_sections.reference_age_display_label(
        "18-24", ["18-19", "22-24", "65-99", "Other", "other"]
    )
    assert label == "25-99"
    assert (
        legacy_bias_sections.reference_age_display_label("freshmen", ["65-99"]) is None
    )

    frame = pd.DataFrame(
        {
            "age_bias_group": ["18-24", "other"],
            "age_group": ["20-21", "Other"],
        }
    )
    labeled = legacy_bias_sections.apply_reference_age_label(
        frame, "age_bias_group", "18-24", ["20-21", "65-99", "Other"]
    )
    assert labeled["age_bias_group"].tolist() == ["18-24", "25-99"]
    assert labeled["age_group"].tolist() == ["20-21", "Other"]
