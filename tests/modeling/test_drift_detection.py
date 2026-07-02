import logging

import pandas as pd
import pytest

from edvise.modeling.drift_detection import (
    compute_numeric_ks_drift,
    log_numeric_ks_drift,
)


def test_compute_numeric_ks_drift_detects_shifted_distribution() -> None:
    df_train = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.3, 0.4]})
    df_infer = pd.DataFrame({"x": [0.6, 0.7, 0.8, 0.9, 1.0]})

    results = compute_numeric_ks_drift(df_train, df_infer)

    assert len(results) == 1
    assert results[0].feature == "x"
    assert results[0].ks_stat == pytest.approx(1.0)
    assert results[0].p_value < 0.05


def test_compute_numeric_ks_drift_ignores_non_numeric_columns() -> None:
    df_train = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
    df_infer = pd.DataFrame({"x": [1, 2, 3], "cat": ["d", "e", "f"]})

    results = compute_numeric_ks_drift(df_train, df_infer)

    assert [r.feature for r in results] == ["x"]
    assert results[0].ks_stat == pytest.approx(0.0)


def test_log_numeric_ks_drift_writes_to_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df_train = pd.DataFrame({"x": [0.0, 0.1, 0.2], "y": [1.0, 1.0, 1.0]})
    df_infer = pd.DataFrame({"x": [0.8, 0.9, 1.0], "y": [1.0, 1.0, 1.0]})
    logger = logging.getLogger("test.drift")

    with caplog.at_level(logging.INFO, logger="test.drift"):
        log_numeric_ks_drift(
            df_train,
            df_infer,
            top_n=2,
            context="unit test",
            logger=logger,
        )

    messages = [record.message for record in caplog.records]
    assert any("unit test" in message for message in messages)
    assert any("x" in message and "KS=" in message for message in messages)
