from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import joblib
import mlflow
from tensorflow import keras

from src.config import MODEL_PATH, MODEL_SCALER_PATH, REPORTS_DIR, load_params, tracking_uri
from src.ingest import fetch_aapl, save_raw
from src.metrics import inverse_close, regression_metrics
from src.model import build_lstm
from src.monitor import generate_drift_report
from src.preprocess import make_windows, prepare_sequences
from src.train import configure_mlflow, log_keras_model


def fetch_and_version_aapl():
    data = fetch_aapl()
    save_raw(data)
    return data


def generate_report_and_check_drift(data) -> bool:
    prepared = prepare_sequences(data)
    reference = prepared["close"][: prepared["split"]]
    current = prepared["close"][-prepared["window"] :]
    return generate_drift_report(reference, current)


def chronological_retrain_split(data, comparison_split: float):
    prepared = prepare_sequences(data)
    n = len(prepared["close"])
    comparison_start = int(n * (1 - comparison_split))
    train_data = data.iloc[:comparison_start]
    comparison_data = data.iloc[comparison_start:]
    return train_data, comparison_data, prepared


def windows_from_frame(frame, scaler, window: int):
    close = frame[["Close"]].dropna().to_numpy(dtype=float)
    if len(close) <= window:
        raise ValueError("Not enough observations to build evaluation windows.")
    scaled = scaler.transform(close)
    features, targets, _ = make_windows(scaled, window)
    return features, targets


def evaluate_rmse(model, scaler, frame, window: int) -> float:
    features, targets = windows_from_frame(frame, scaler, window)
    pred = inverse_close(scaler, model.predict(features, verbose=0))
    actual = inverse_close(scaler, targets)
    return regression_metrics(actual, pred)["rmse"]


def train_and_log_candidate(train_data, params: dict):
    prepared = prepare_sequences(train_data, params)
    model_cfg = params["model"]
    model = build_lstm(
        prepared["window"], int(model_cfg["lstm_units"]), float(model_cfg["dropout"])
    )
    history = model.fit(
        prepared["X_train"],
        prepared["y_train"],
        epochs=int(model_cfg["epochs"]),
        batch_size=int(model_cfg["batch_size"]),
        validation_split=float(model_cfg["validation_split"]),
        shuffle=False,
        verbose=1,
    )
    configure_mlflow(params)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "window": prepared["window"],
                "epochs": model_cfg["epochs"],
                "units": model_cfg["lstm_units"],
                "batch_size": model_cfg["batch_size"],
                "purpose": "retrain_candidate",
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            scaler_file = Path(tmp) / "scaler.joblib"
            joblib.dump(prepared["scaler"], scaler_file)
            mlflow.log_artifact(str(scaler_file))
        model_info = log_keras_model(model)
        mlflow.log_metrics({"train_loss": float(history.history["loss"][-1])})
        return {
            "model": model,
            "scaler": prepared["scaler"],
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
        }


def load_champion_with_scaler(params: dict):
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.MlflowClient()
    name = params["mlflow"]["registered_model"]
    alias = params["mlflow"]["alias"]
    try:
        version = client.get_model_version_by_alias(name, alias)
        model = mlflow.keras.load_model(f"models:/{name}/{version.version}")
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="scaler.joblib"
        )
        scaler = joblib.load(scaler_path)
        return model, scaler, version.version
    except Exception:
        if MODEL_PATH.exists() and MODEL_SCALER_PATH.exists():
            return keras.models.load_model(MODEL_PATH), joblib.load(MODEL_SCALER_PATH), "local"
        return None, None, None


def register_and_promote(candidate: dict, params: dict) -> str:
    registered = mlflow.register_model(
        model_uri=candidate["model_uri"],
        name=params["mlflow"]["registered_model"],
    )
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        params["mlflow"]["registered_model"],
        params["mlflow"]["alias"],
        registered.version,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate["model"].save(MODEL_PATH)
    joblib.dump(candidate["scaler"], MODEL_SCALER_PATH)
    return str(registered.version)


def retrain(scheduled: bool = False) -> str:
    params = load_params()
    data = fetch_and_version_aapl()
    drift_detected = generate_report_and_check_drift(data)
    if not (drift_detected or scheduled):
        message = "No retraining"
        print(message)
        return message
    train_data, comparison_data, _prepared = chronological_retrain_split(
        data, float(params["retrain"]["comparison_split"])
    )
    candidate = train_and_log_candidate(train_data, params)
    current_model, current_scaler, current_version = load_champion_with_scaler(params)
    window = int(params["preprocess"]["window"])
    new_rmse = evaluate_rmse(candidate["model"], candidate["scaler"], comparison_data, window)
    decision = {
        "drift_detected": drift_detected,
        "scheduled": scheduled,
        "comparison_start": str(comparison_data.index.min().date()),
        "comparison_end": str(comparison_data.index.max().date()),
        "candidate_rmse": new_rmse,
    }
    if current_model is None:
        version = register_and_promote(candidate, params)
        decision["decision"] = "Candidate promoted"
        decision["reason"] = "No current champion"
        decision["promoted_version"] = version
    else:
        old_rmse = evaluate_rmse(current_model, current_scaler, comparison_data, window)
        decision["current_rmse"] = old_rmse
        decision["current_version"] = current_version
        if new_rmse < old_rmse:
            version = register_and_promote(candidate, params)
            decision["decision"] = "Candidate promoted"
            decision["promoted_version"] = version
        else:
            decision["decision"] = "Current model retained"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrain_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))
    return decision["decision"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the AAPL LSTM when drift or schedule requires it."
    )
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Retrain even when drift is not detected.",
    )
    args = parser.parse_args()
    retrain(scheduled=args.scheduled)


if __name__ == "__main__":
    main()
