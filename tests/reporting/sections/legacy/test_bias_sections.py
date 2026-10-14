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


def test_bias_groups_section_with_valid_aliases(mock_card, registry):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "First-Generation Status",
        "gender": "Gender",
        "ethnicity_ipeds": "Ethnicity",
        "demo_race": "Race",
    }

    legacy_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]
    print(result)

    assert (
        "- Our assessment for FNR Parity was conducted across the following student groups."
        in result
    )
    assert "- First-Generation Status" in result
    assert "- Gender" in result
    assert "- Ethnicity" in result
    assert "- Race" in result
    assert "pre-mitigation" not in result
    assert "Support Label" not in result


def test_bias_groups_section_with_aliases_that_need_friendlycase(mock_card):
    mock_card.cfg.student_group_aliases = {
        "firstgenflag": "first_generation_status",
        "disabilityflag": "disability_status",
    }

    registry = SectionRegistry()
    legacy_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    result = rendered["bias_groups_section"]
    print(result)

    assert "- First Generation Status" in result
    assert "- Disability Status" in result


def test_bias_groups_section_with_missing_aliases(mock_card, caplog):
    mock_card.cfg.student_group_aliases = None
    mock_card.format.friendly_case.side_effect = lambda x: x.replace("_", " ").title()

    registry = SectionRegistry()
    legacy_bias_sections.register_bias_sections(mock_card, registry)

    with caplog.at_level("WARNING"):
        rendered = registry.render_all()

    result = rendered["bias_groups_section"]
    assert "- Unable to extract student groups" in result
    assert any(
        "Failed to extract student groups" in message for message in caplog.messages
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
    assert "Pre-Mitigation" not in result
    assert "Post-Mitigation" not in result
    assert "Support Label" not in result
    assert "Post-mitigation metrics were not logged." not in result


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


@patch("edvise.reporting.utils.utils.download_artifact")
def test_bias_summary_with_mitigation_includes_logged_fnr(
    mock_download_artifact, mock_card, tmp_path
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

    assert "####Pre-Mitigation Disparities by Student Group" in summary
    assert "pre-mitigation training audit at the default threshold" in summary
    assert "####Disparities by Student Group" not in summary
    assert "Age Group" in summary
    assert "12% difference" in summary
    assert "####Post-Mitigation Dual Threshold" in summary
    assert "**Age Group** (`agegroup`)" in summary
    assert "**25 and over**" in summary
    assert "**0.35**" in summary
    assert "**0.5**" in summary
    assert "Support scores stay the same." in summary
    assert "target group versus everyone else" in summary
    assert "**12.0%**" in summary
    assert "**20.0%**" in summary
    assert "**8.0%**" in summary
    assert "Post-mitigation metrics were not logged." not in summary
    assert (
        "Advisors see this dual-threshold decision via Support Label when the pipeline writes one."
        in summary
    )
    assert "pre-mitigation training audit at the default threshold" in groups
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

    assert "####Pre-Mitigation Disparities by Student Group" in summary
    assert "Gender" in summary
    assert "12% difference" in summary
    assert "####Post-Mitigation Dual Threshold" in summary
    assert "**Pell Status** (`pell_flag`)" in summary
    assert "**recipient**" in summary
    assert "**0.4**" in summary
    assert "**0.5**" in summary
    assert "Post-mitigation metrics were not logged." in summary
    assert "Target group FNR" not in summary
    assert (
        "Advisors see this dual-threshold decision via Support Label when the pipeline writes one."
        in summary
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
    assert "**Student Group** (`student_group`)" in summary
    assert "Post-mitigation metrics were not logged." in summary
    assert "12% difference" in summary


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
            "group": ["18-24", "18-24", "18-24"],
            "threshold": [0.2, 0.3, 0.4],
            "reference_fnr": [0.21621621621621623] * 3,
            "group_fnr": [0.05, 0.2074074074074074, 0.29074074074074074],
            "fnr_abs_gap": [
                0.16621621621621624,
                0.008808808808808827,
                0.07452452452452452,
            ],
            "within_tolerance": [False, True, False],
            "balanced_accuracy": [0.64, 0.69, 0.74],
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
    assert "**0.9%**" in summary
    assert "Post-mitigation metrics were not logged." not in summary
    mock_list_paths.assert_called_once_with("run123", "bias_mitigation")


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

    assert "`agegroup`" in summary
    assert "**Student Group**" not in summary
    assert "**0.55**" in summary
    assert "**0.3**" in summary
    assert "Post-mitigation metrics were not logged." in summary
    assert "Post-Mitigation Disparities by Student Group" not in summary


def test_bias_summary_shows_post_mitigation_group_plots(mock_card, tmp_path):
    mock_card.assets_folder = str(tmp_path)
    mock_card.cfg.student_group_cols = ["gender", "ethnicity"]
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
                "group": "age_bias_group",
                "split_name": "test",
                "subgroups": "18-24 vs 25+",
                "fnr_percentage_difference": 0.2,
                "type": "a significant difference",
            }
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
            }
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

    assert "Pre-Mitigation Disparities by Student Group" in summary
    assert "20% difference" in summary
    assert "Post-Mitigation Disparities by Student Group" in summary
    assert (
        "18-24 students are flagged at 0.3. Everyone else is flagged at 0.5." in summary
    )
    assert "#####Age Bias Group" in summary
    assert "**other** FNR: **21.6%**" in summary
    assert "**18-24** FNR: **20.7%**" in summary
    assert (
        "No statistically significant disparity was found for this group after the dual threshold."
        not in summary
    )
    assert str(age_plot) in summary
    assert "#####Gender" in summary
    assert "**Male** FNR: **23.0%**" in summary
    assert "**Female** FNR: **19.5%**" in summary
    assert "**99.0%**" not in summary
    assert "12% difference" in summary
    assert str(gender_plot) in summary
    assert "ethnicity" not in summary.lower()
    assert "Post-mitigation metrics were not logged." in summary
