from __future__ import annotations

import json
from functools import lru_cache

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import (
    DRIFT_REPORT_PATH,
    HISTORY_PATH,
    LOSS_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_SCALER_PATH,
    PRED_PLOT_PATH,
    RAW_CSV,
    REPORTS_DIR,
    ROOT,
    load_params,
    tracking_uri,
)


class PriceWindow(BaseModel):
    close: list[float] = Field(min_length=60, max_length=60)


class PredictionResponse(BaseModel):
    predicted_close: float
    model_version: str


@lru_cache(maxsize=1)
def load_serving_assets():
    params = load_params()
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
        return model, scaler, str(version.version)
    except Exception:
        if MODEL_PATH.exists() and MODEL_SCALER_PATH.exists():
            from tensorflow import keras

            model = keras.models.load_model(MODEL_PATH)
            scaler = joblib.load(MODEL_SCALER_PATH)
            return model, scaler, "local"
        raise RuntimeError("No registered champion or local model artifacts were found.")


app = FastAPI(
    title="AAPL Next-Day Close Prediction API",
    description="Serves the champion LSTM registered in MLflow.",
    version="1.0.0",
)


@app.get("/health")
def health():
    try:
        _, _, version = load_serving_assets()
        return {"status": "ok", "model_version": version}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}


@app.get("/api/dashboard")
def dashboard_data():
    if not RAW_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail="Market data is not available. Run python -m src.ingest first.",
        )

    try:
        market_data = pd.read_csv(RAW_CSV, parse_dates=["Date"])
        if "Close" not in market_data or "Date" not in market_data:
            raise ValueError("The AAPL CSV must include Date and Close columns.")
        market_data = market_data.dropna(subset=["Date", "Close"]).sort_values("Date")
        if market_data.empty:
            raise ValueError("The AAPL CSV contains no usable closing prices.")
        market_data["Close"] = pd.to_numeric(market_data["Close"], errors="raise")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read AAPL market data: {exc}"
        ) from exc

    recent = market_data.tail(252)
    latest = recent.iloc[-1]
    metrics = None
    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not read evaluation metrics: {exc}"
            ) from exc

    history = None
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not read model training history: {exc}"
            ) from exc

    stages = {
        "ingest": RAW_CSV.exists(),
        "preprocess": (ROOT / "data" / "processed" / "sequences.npz").exists(),
        "train": MODEL_PATH.exists() and MODEL_SCALER_PATH.exists(),
        "evaluate": METRICS_PATH.exists(),
        "monitor": DRIFT_REPORT_PATH.exists(),
    }
    return {
        "ticker": "AAPL",
        "observations": int(len(market_data)),
        "latest_date": latest["Date"].date().isoformat(),
        "latest_close": float(latest["Close"]),
        "points": [
            {"date": row.Date.date().isoformat(), "close": float(row.Close)}
            for row in recent.itertuples(index=False)
        ],
        "window": [float(value) for value in market_data["Close"].tail(60)],
        "metrics": metrics,
        "history": history,
        "pipeline": {
            "dvc_initialized": (ROOT / ".dvc" / "config").exists(),
            "stages": stages,
        },
        "reports": {
            "prediction_plot": PRED_PLOT_PATH.exists(),
            "loss_plot": LOSS_PLOT_PATH.exists(),
            "drift_report": DRIFT_REPORT_PATH.exists(),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PriceWindow):
    try:
        model, scaler, version = load_serving_assets()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    window = np.array(payload.close, dtype=float).reshape(-1, 1)
    scaled = scaler.transform(window)
    scaled_prediction = model.predict(scaled.reshape(1, 60, 1), verbose=0)
    price = scaler.inverse_transform(scaled_prediction)[0, 0]
    return PredictionResponse(predicted_close=float(price), model_version=version)


app.mount("/reports", StaticFiles(directory=REPORTS_DIR, check_dir=False), name="reports")
app.mount(
    "/static",
    StaticFiles(directory=ROOT / "src" / "static", check_dir=False),
    name="static",
)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "src" / "static" / "index.html")
