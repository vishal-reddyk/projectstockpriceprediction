from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import mlflow
from tensorflow import keras

from src.api import TICKERS, load_market_data
from src.config import HISTORY_PATH, METRICS_PATH, MODELS_DIR, load_params
from src.metrics import inverse_close, regression_metrics
from src.model import build_lstm
from src.preprocess import prepare_sequences
from src.train import log_keras_model


def train_ticker(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise ValueError(f"Unsupported ticker {ticker}. Choose from {', '.join(TICKERS)}.")

    params = load_params()
    market_data = load_market_data(ticker).set_index("Date")
    prepared = prepare_sequences(market_data, params)
    if not len(prepared["X_train"]) or not len(prepared["X_test"]):
        raise ValueError(f"Not enough observations to train and evaluate {ticker}.")

    model_config = params["model"]
    model = build_lstm(
        prepared["window"],
        int(model_config["lstm_units"]),
        float(model_config["dropout"]),
    )
    history = model.fit(
        prepared["X_train"],
        prepared["y_train"],
        epochs=int(model_config["epochs"]),
        batch_size=int(model_config["batch_size"]),
        validation_split=float(model_config["validation_split"]),
        shuffle=False,
        verbose=1,
    )

    predictions = model.predict(prepared["X_test"], verbose=0)
    predicted_close = inverse_close(prepared["scaler"], predictions)
    actual_close = inverse_close(prepared["scaler"], prepared["y_test"])
    metrics = regression_metrics(actual_close, predicted_close)
    ticker_dir = (
        MODELS_DIR if ticker == "AAPL" else MODELS_DIR / "tickers" / ticker
    )
    ticker_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{ticker.lower()}_lstm"
    registered_name = f"{ticker}_LSTM"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{ticker} five-year training") as run:
        mlflow.log_params(
            {
                "ticker": ticker,
                "window": prepared["window"],
                "epochs": model_config["epochs"],
                "units": model_config["lstm_units"],
                "batch_size": model_config["batch_size"],
                "dropout": model_config["dropout"],
                "train_sequences": len(prepared["X_train"]),
                "test_sequences": len(prepared["X_test"]),
            }
        )
        mlflow.log_metrics(metrics)
        history_payload = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        history_path = HISTORY_PATH if ticker == "AAPL" else ticker_dir / "history.json"
        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(history_path))
        with tempfile.TemporaryDirectory() as temp_dir:
            scaler_path = Path(temp_dir) / "scaler.joblib"
            joblib.dump(prepared["scaler"], scaler_path)
            mlflow.log_artifact(str(scaler_path))
        model_info = log_keras_model(model)
        registered = mlflow.register_model(model_info.model_uri, registered_name)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            registered_name, params["mlflow"]["alias"], registered.version
        )

    model_path = MODELS_DIR / "lstm.keras" if ticker == "AAPL" else ticker_dir / "lstm.keras"
    scaler_path = (
        MODELS_DIR / "scaler.joblib"
        if ticker == "AAPL"
        else ticker_dir / "scaler.joblib"
    )
    model.save(model_path)
    joblib.dump(prepared["scaler"], scaler_path)
    result = {
        **metrics,
        "ticker": ticker,
        "model_version": str(registered.version),
        "experiment": experiment_name,
        "registered_model": registered_name,
        "run_id": run.info.run_id,
        "test_sequences": int(len(actual_close)),
        "start_date": str(prepared["dates"][0].astype("datetime64[D]")),
        "end_date": str(prepared["dates"][-1].astype("datetime64[D]")),
    }
    metrics_path = METRICS_PATH if ticker == "AAPL" else ticker_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    keras.backend.clear_session()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and register a separate next-day price model per ticker."
    )
    parser.add_argument("--ticker", choices=sorted(TICKERS), help="Train one ticker.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train every configured ticker, including tickers with an existing model.",
    )
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    elif args.all:
        tickers = list(TICKERS)
    else:
        tickers = []
        for ticker in TICKERS:
            model_path = (
                MODELS_DIR / "lstm.keras"
                if ticker == "AAPL"
                else MODELS_DIR / "tickers" / ticker / "lstm.keras"
            )
            if not model_path.exists():
                tickers.append(ticker)

    if not tickers:
        print("All configured tickers already have a trained model.")
        return

    failures: list[str] = []
    for ticker in tickers:
        print(f"\nTraining {ticker}...", flush=True)
        try:
            train_ticker(ticker)
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            print(f"Training failed for {ticker}: {exc}", flush=True)

    if failures:
        raise RuntimeError("One or more ticker models failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
