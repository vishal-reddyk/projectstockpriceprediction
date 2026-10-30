from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    PROCESSED_DIR,
    RAW_CSV,
    SCALER_PATH,
    SEQUENCES_PATH,
    load_params,
)


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    data = data.sort_index()
    if data.empty:
        raise ValueError("Raw CSV is empty.")
    return data


def close_series(data: pd.DataFrame) -> pd.Series:
    return data.sort_index()["Close"].dropna().astype(float)


def close_array(data: pd.DataFrame) -> np.ndarray:
    return close_series(data).to_numpy().reshape(-1, 1)


def make_windows(
    scaled: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, target_index = [], [], []
    for i in range(window, len(scaled)):
        features.append(scaled[i - window : i])
        targets.append(scaled[i, 0])
        target_index.append(i)
    return np.array(features), np.array(targets), np.array(target_index)


def split_by_target(
    features: np.ndarray,
    targets: np.ndarray,
    target_index: np.ndarray,
    split: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = target_index < split
    test_mask = target_index >= split
    return features[train_mask], targets[train_mask], features[test_mask], targets[test_mask]


def prepare_sequences(data: pd.DataFrame, params: dict | None = None) -> dict:
    settings = (params or load_params())["preprocess"]
    window = int(settings["window"])
    train_split = float(settings["train_split"])
    series = close_series(data)
    close = series.to_numpy().reshape(-1, 1)
    split = int(len(close) * train_split)
    scaler = MinMaxScaler()
    scaler.fit(close[:split])
    scaled = scaler.transform(close)
    features, targets, target_index = make_windows(scaled, window)
    x_train, y_train, x_test, y_test = split_by_target(
        features, targets, target_index, split
    )
    return {
        "scaler": scaler,
        "close": close,
        "split": split,
        "window": window,
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "dates": series.index.to_numpy(),
        "target_index": target_index,
    }


def save_processed(prepared: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEQUENCES_PATH,
        X_train=prepared["X_train"],
        y_train=prepared["y_train"],
        X_test=prepared["X_test"],
        y_test=prepared["y_test"],
        close=prepared["close"],
        target_index=prepared["target_index"],
        dates=prepared["dates"].astype("datetime64[ns]").astype("int64"),
        split=np.array([prepared["split"]]),
        window=np.array([prepared["window"]]),
    )
    joblib.dump(prepared["scaler"], SCALER_PATH)
    summary = {
        "rows": int(len(prepared["close"])),
        "split": int(prepared["split"]),
        "window": int(prepared["window"]),
        "train_sequences": int(len(prepared["X_train"])),
        "test_sequences": int(len(prepared["X_test"])),
        "start_date": str(pd.Timestamp(prepared["dates"][0]).date()),
        "end_date": str(pd.Timestamp(prepared["dates"][-1]).date()),
    }
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    prepared = prepare_sequences(load_raw())
    save_processed(prepared)


if __name__ == "__main__":
    main()
