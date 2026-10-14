from unittest.mock import MagicMock, patch

from edvise.configs.es import ESProjectConfig
from edvise.reporting.sections.es import register_sections
from edvise.reporting.sections.registry import SectionRegistry


def make_es_config(
    student_group_cols: list[str] | None = None,
) -> ESProjectConfig:
    kwargs: dict = {
        "institution_id": "inst_id",
        "institution_name": "Inst Name",
        "model": {"experiment_id": "exp123", "run_id": "abc"},
        "datasets": {
            "raw_course": "dummy.csv",
            "raw_cohort": "dummy.csv",
        },
        "preprocessing": {
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
        "modeling": {
            "feature_selection": {
                "collinear_threshold": 10.0,
                "low_variance_threshold": 0.0,
                "incomplete_threshold": 0.5,
            },
            "training": {"primary_metric": "logloss", "timeout_minutes": 10},
        },
        "split_col": None,
    }
    if student_group_cols is not None:
        kwargs["student_group_cols"] = student_group_cols
    return ESProjectConfig(**kwargs)


@patch(
    "edvise.reporting.sections.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=["gender", "race"],
)
def test_bias_groups_section_lists_only_assessed_groups(mock_groups):
    """Card follows run artifacts, not the full configured student_group_cols."""
    card = MagicMock()
    card.cfg = make_es_config()  # default includes status fields
    card.run_id = "run123"
    card.format.friendly_case.side_effect = lambda s: s.replace("_", " ").title()
    card.format.indent_level.side_effect = lambda n: "  " * n

    registry = SectionRegistry()
    register_sections(card, registry)
    sections = registry.render_all()

    bias_section = sections["bias_groups_section"]
    assert "Gender" in bias_section
    assert "Race" in bias_section
    assert "Disability Status" not in bias_section
    assert "Incarcerated Status" not in bias_section
    assert "Military Status" not in bias_section
    assert "Employment Status" not in bias_section
    assert "Learner Age" not in bias_section
    mock_groups.assert_called_once_with("run123")


@patch(
    "edvise.reporting.sections.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=[],
)
def test_bias_groups_section_handles_missing_artifacts(mock_groups):
    card = MagicMock()
    card.cfg = make_es_config()
    card.run_id = "run123"
    card.format.friendly_case.side_effect = lambda s: s.replace("_", " ").title()
    card.format.indent_level.side_effect = lambda n: "  " * n

    registry = SectionRegistry()
    register_sections(card, registry)
    sections = registry.render_all()

    assert (
        "Unable to determine assessed student groups" in sections["bias_groups_section"]
    )
