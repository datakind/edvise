from edvise.configs.es import ESProjectConfig


def test_es_default_student_group_cols_use_edvise_column_names() -> None:
    cfg = ESProjectConfig(
        institution_id="test_inst",
        institution_name="Test Institution",
        datasets={"raw_course": "course.csv", "raw_cohort": "cohort.csv"},
    )
    assert cfg.student_group_cols == [
        "learner_age",
        "race",
        "ethnicity",
        "gender",
        "first_generation_status",
        "incarcerated_status",
        "military_status",
        "employment_status",
        "disability_status",
    ]


def test_es_non_feature_cols_include_student_group_cols() -> None:
    cfg = ESProjectConfig(
        institution_id="test_inst",
        institution_name="Test Institution",
        datasets={"raw_course": "course.csv", "raw_cohort": "cohort.csv"},
    )
    assert "learner_age" in cfg.non_feature_cols
    assert "first_generation_status" in cfg.non_feature_cols
    assert "incarcerated_status" in cfg.non_feature_cols
    assert "military_status" in cfg.non_feature_cols
    assert "employment_status" in cfg.non_feature_cols
    assert "disability_status" in cfg.non_feature_cols
    assert "student_age" not in cfg.non_feature_cols
    assert "first_gen" not in cfg.non_feature_cols


def test_es_non_feature_cols_protect_default_bias_vars_despite_override() -> None:
    """Shorter student_group_cols must not allow status fields to become features."""
    cfg = ESProjectConfig(
        institution_id="test_inst",
        institution_name="Test Institution",
        datasets={"raw_course": "course.csv", "raw_cohort": "cohort.csv"},
        student_group_cols=["gender", "race"],
    )
    assert cfg.student_group_cols == ["gender", "race"]
    for col in (
        "learner_age",
        "ethnicity",
        "first_generation_status",
        "incarcerated_status",
        "military_status",
        "employment_status",
        "disability_status",
    ):
        assert col in cfg.non_feature_cols
