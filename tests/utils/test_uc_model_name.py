"""Tests for Unity Catalog model-name encode/decode helpers."""

import pytest

from edvise.utils.uc_model_name import decode_uc_model_name, encode_uc_model_name


@pytest.mark.parametrize(
    ("display", "uc"),
    [
        (
            "graduation_in_3y_ft_4.5y_pt_checkpoint_30_credits",
            "graduation_in_3y_ft_4d5y_pt_checkpoint_30_credits",
        ),
        (
            "graduation_in_3y_ft_4.5Y_pt_checkpoint_30_credits",
            "graduation_in_3y_ft_4d5y_pt_checkpoint_30_credits",
        ),
        ("retention_into_year_2", "retention_into_year_2"),
        ("sample_model_for_school_1", "sample_model_for_school_1"),
        # Unrelated dots (not time-limit units) are left alone.
        ("model.v2", "model.v2"),
        ("1.2.csv", "1.2.csv"),
    ],
)
def test_encode_uc_model_name(display: str, uc: str) -> None:
    assert encode_uc_model_name(display) == uc


@pytest.mark.parametrize(
    ("uc", "display"),
    [
        (
            "graduation_in_3y_ft_4d5y_pt_checkpoint_30_credits",
            "graduation_in_3y_ft_4.5Y_pt_checkpoint_30_credits",
        ),
        (
            "graduation_in_3y_ft_4d5Y_pt_checkpoint_30_credits",
            "graduation_in_3y_ft_4.5Y_pt_checkpoint_30_credits",
        ),
        ("retention_into_year_2", "retention_into_year_2"),
        # Already-display form keeps the value and uppercases the unit.
        (
            "graduation_in_3y_ft_4.5y_pt_checkpoint_30_credits",
            "graduation_in_3y_ft_4.5Y_pt_checkpoint_30_credits",
        ),
        # Placeholder without a time-limit unit is left alone.
        ("id1d2suffix", "id1d2suffix"),
    ],
)
def test_decode_uc_model_name(uc: str, display: str) -> None:
    assert decode_uc_model_name(uc) == display


def test_encode_decode_roundtrip() -> None:
    uc = "graduation_in_3y_ft_4d5y_pt_checkpoint_30_credits"
    display = "graduation_in_3y_ft_4.5Y_pt_checkpoint_30_credits"
    assert encode_uc_model_name(display) == uc
    assert decode_uc_model_name(uc) == display
