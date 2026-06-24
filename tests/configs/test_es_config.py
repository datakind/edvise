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
    ]


def test_es_non_feature_cols_include_student_group_cols() -> None:
    cfg = ESProjectConfig(
        institution_id="test_inst",
        institution_name="Test Institution",
        datasets={"raw_course": "course.csv", "raw_cohort": "cohort.csv"},
    )
    assert "learner_age" in cfg.non_feature_cols
    assert "first_generation_status" in cfg.non_feature_cols
    assert "student_age" not in cfg.non_feature_cols
    assert "first_gen" not in cfg.non_feature_cols
