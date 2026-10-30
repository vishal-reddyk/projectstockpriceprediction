import numpy as np
import pandas as pd

from src.metrics import inverse_close
from src.preprocess import prepare_sequences


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values)


def test_inverse_close_flattens_values():
    result = inverse_close(_IdentityScaler(), [[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_prepare_sequences_is_chronological():
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    close = np.linspace(100, 160, len(dates))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    frame.index.name = "Date"
    prepared = prepare_sequences(
        frame,
        {
            "preprocess": {"window": 60, "train_split": 0.8},
        },
    )
    assert prepared["X_train"].shape[1:] == (60, 1)
    assert prepared["split"] == 96
    assert len(prepared["X_test"]) > 0
    assert prepared["target_index"][0] == 60
