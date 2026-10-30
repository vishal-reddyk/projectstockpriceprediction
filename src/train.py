from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np

from src.config import (
    HISTORY_PATH,
    MODEL_PATH,
    MODEL_SCALER_PATH,
    SCALER_PATH,
    SEQUENCES_PATH,
    load_params,
    tracking_uri,
)
from src.metrics import inverse_close, regression_metrics
from src.model import build_lstm


def configure_mlflow(params: dict) -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(params["mlflow"]["experiment"])


def load_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    arrays = np.load(SEQUENCES_PATH, allow_pickle=False)
    scaler = joblib.load(SCALER_PATH)
    return arrays["X_train"], arrays["y_train"], arrays["X_test"], arrays["y_test"], scaler


def log_keras_model(model):
    try:
        return mlflow.keras.log_model(model, name="model")
    except TypeError:
        return mlflow.keras.log_model(model, artifact_path="model")


def train(params: dict | None = None):
    params = params or load_params()
    configure_mlflow(params)
    x_train, y_train, x_test, y_test, scaler = load_training_arrays()
    model_cfg = params["model"]
    window = int(params["preprocess"]["window"])
    model = build_lstm(window, int(model_cfg["lstm_units"]), float(model_cfg["dropout"]))
    history = model.fit(
        x_train,
        y_train,
        epochs=int(model_cfg["epochs"]),
        batch_size=int(model_cfg["batch_size"]),
        validation_split=float(model_cfg["validation_split"]),
        shuffle=False,
        verbose=1,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, MODEL_SCALER_PATH)
    history_payload = {key: [float(v) for v in values] for key, values in history.history.items()}
    HISTORY_PATH.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    pred = inverse_close(scaler, model.predict(x_test, verbose=0))
    actual = inverse_close(scaler, y_test)
    metrics = regression_metrics(actual, pred)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "window": window,
                "epochs": model_cfg["epochs"],
                "units": model_cfg["lstm_units"],
                "batch_size": model_cfg["batch_size"],
                "dropout": model_cfg["dropout"],
            }
        )
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            scaler_file = Path(tmp) / "scaler.joblib"
            joblib.dump(scaler, scaler_file)
            mlflow.log_artifact(str(scaler_file))
        model_info = log_keras_model(model)
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=params["mlflow"]["registered_model"],
        )
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            params["mlflow"]["registered_model"],
            params["mlflow"]["alias"],
            registered.version,
        )
        print(f"Run ID: {run.info.run_id}")
        print(f"Registered {params['mlflow']['registered_model']} v{registered.version}")
        print(json.dumps(metrics, indent=2))
        return {
            "model": model,
            "scaler": scaler,
            "metrics": metrics,
            "run_id": run.info.run_id,
            "version": registered.version,
            "history": history_payload,
        }


def main() -> None:
    train()


if __name__ == "__main__":
    main()
