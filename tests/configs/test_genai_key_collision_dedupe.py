"""KeyCollisionDedupeConfig nested under DatasetConfig."""

from edvise.configs.genai import DatasetConfig, KeyCollisionDedupeConfig


def test_dataset_config_optional_dedupe_none():
    d = DatasetConfig(
        primary_keys=["a"],
        files=["/tmp/x.csv"],
    )
    assert d.dedupe is None


def test_dataset_config_with_dedupe():
    d = DatasetConfig(
        primary_keys=["student_id", "term", "class_number"],
        files=["/tmp/course.csv"],
        dedupe=KeyCollisionDedupeConfig(
            key_cols=["student_id", "term", "class_number"],
            conflict_columns=["course_classification"],
            disambiguate_column="class_number",
        ),
    )
    assert d.dedupe.disambiguate_sep == "-"
    assert d.dedupe.conflict_columns == ["course_classification"]
