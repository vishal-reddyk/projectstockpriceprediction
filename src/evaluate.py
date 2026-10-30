from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import (
    HISTORY_PATH,
    LOSS_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_SCALER_PATH,
    PRED_PLOT_PATH,
    REPORTS_DIR,
    SEQUENCES_PATH,
)
from src.metrics import inverse_close, regression_metrics


def load_assets():
    arrays = np.load(SEQUENCES_PATH, allow_pickle=False)
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(MODEL_SCALER_PATH)
    return arrays, model, scaler


def plot_predictions(dates, actual, predicted, path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual close")
    plt.plot(dates, predicted, label="Predicted close")
    plt.title("AAPL test set: actual vs predicted close")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_loss(history: dict, path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.get("loss", []), label="Training loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation loss")
    plt.title("LSTM training and validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def evaluate() -> dict:
    arrays, model, scaler = load_assets()
    pred = inverse_close(scaler, model.predict(arrays["X_test"], verbose=0))
    actual = inverse_close(scaler, arrays["y_test"])
    metrics = regression_metrics(actual, pred)
    dates_ns = arrays["dates"].astype("datetime64[ns]")
    test_target_pos = arrays["target_index"][arrays["target_index"] >= arrays["split"][0]]
    test_dates = pd.to_datetime(dates_ns[test_target_pos])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_predictions(test_dates, actual, pred, PRED_PLOT_PATH)
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        plot_loss(history, LOSS_PLOT_PATH)
    payload = {
        **metrics,
        "test_sequences": int(len(actual)),
        "start_date": str(test_dates.min().date()) if len(test_dates) else None,
        "end_date": str(test_dates.max().date()) if len(test_dates) else None,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
