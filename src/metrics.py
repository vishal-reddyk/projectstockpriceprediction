from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    rmse = float(mean_squared_error(actual, predicted) ** 0.5)
    mae = float(mean_absolute_error(actual, predicted))
    mape = float(100 * mean_absolute_percentage_error(actual, predicted))
    return {"rmse": rmse, "mae": mae, "mape_percent": mape}


def inverse_close(scaler, values) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(array).ravel()
