import numpy as np
import pandas as pd
import pytest

from src.ingest import validate_ohlcv
from src.metrics import regression_metrics
from src.preprocess import make_windows, split_by_target


def test_make_windows_shapes():
    scaled = np.arange(10, dtype=float).reshape(-1, 1)
    features, targets, index = make_windows(scaled, window=3)
    assert features.shape == (7, 3, 1)
    assert targets.shape == (7,)
    assert index.tolist() == list(range(3, 10))
    np.testing.assert_array_equal(features[0].ravel(), [0, 1, 2])
    assert targets[0] == 3


def test_split_keeps_target_dates_in_test():
    features = np.arange(8).reshape(8, 1, 1).astype(float)
    targets = np.arange(8).astype(float)
    target_index = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    x_train, y_train, x_test, y_test = split_by_target(features, targets, target_index, split=6)
    assert len(x_train) == 4
    assert len(x_test) == 4
    np.testing.assert_array_equal(y_test, [4, 5, 6, 7])


def test_validate_rejects_duplicates():
    idx = pd.to_datetime(["2024-01-01", "2024-01-01"])
    frame = pd.DataFrame(
        {"Open": [1, 1], "High": [1, 1], "Low": [1, 1], "Close": [1, 1], "Volume": [1, 1]},
        index=idx,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_ohlcv(frame)


def test_regression_metrics_perfect_score():
    values = np.array([10.0, 20.0, 30.0])
    metrics = regression_metrics(values, values)
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["mape_percent"] == 0
