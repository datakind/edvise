import pandas as pd
import pytest

from edvise.model_prep.cleanup_features import (
    PDPCleanup,
    target_type_from_config,
)


class _Cfg:
    def __init__(self, target_type: str | None) -> None:
        if target_type is None:
            self.preprocessing = None
        else:
            self.preprocessing = type(
                "Prep",
                (),
                {"target": type("Target", (), {"type_": target_type})()},
            )()


@pytest.mark.parametrize(
    ("target_type", "keep_first_term"),
    [
        ("retention", False),
        ("graduation", True),
        ("credits_earned", True),
        (None, True),
    ],
)
def test_pdp_cleanup_drops_first_term_intensity_only_for_retention(
    target_type: str | None, keep_first_term: bool
) -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"],
            "enrollment_intensity_first_term": ["FULL-TIME"],
            "student_term_enrollment_intensity": ["FULL-TIME"],
        }
    )
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(
        df, target_type=target_type
    )
    assert "student_term_enrollment_intensity" in cleaned.columns
    assert ("enrollment_intensity_first_term" in cleaned.columns) is keep_first_term


def test_target_type_from_config() -> None:
    assert target_type_from_config(_Cfg("retention")) == "retention"
    assert target_type_from_config(_Cfg("graduation")) == "graduation"
    assert target_type_from_config(_Cfg(None)) is None
