from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

import joblib
import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from yfinance.exceptions import YFException

from src.config import (
    DRIFT_REPORT_PATH,
    HISTORY_PATH,
    LOSS_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_SCALER_PATH,
    MODELS_DIR,
    PRED_PLOT_PATH,
    RAW_CSV,
    REPORTS_DIR,
    ROOT,
    load_params,
    tracking_uri,
)


class PriceWindow(BaseModel):
    close: list[float] = Field(min_length=60, max_length=60)
    ticker: str = "AAPL"


class PredictionResponse(BaseModel):
    predicted_close: float
    model_version: str


@lru_cache(maxsize=16)
def load_serving_assets(ticker: str = "AAPL"):
    ticker = ticker.upper()
    if ticker != "AAPL":
        ticker_dir = MODELS_DIR / "tickers" / ticker
        model_path = ticker_dir / "lstm.keras"
        scaler_path = ticker_dir / "scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise RuntimeError(f"No trained prediction model is available for {ticker}.")
        from tensorflow import keras

        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        metrics_path = ticker_dir / "metrics.json"
        version = "local"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            version = str(metrics.get("model_version", version))
        return model, scaler, version

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


TICKERS = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
}


def load_market_data(ticker: str) -> pd.DataFrame:
    if ticker == "AAPL":
        data = pd.read_csv(RAW_CSV, parse_dates=["Date"])
        if "Date" not in data or "Close" not in data:
            raise ValueError("The AAPL CSV must include Date and Close columns.")
        data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    else:
        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            multi_level_index=False,
            progress=False,
        )
        if data.empty:
            raise ValueError(f"No market data was returned for {ticker}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [column[0] for column in data.columns]
        data = data.reset_index()
        if "Date" not in data and "Datetime" in data:
            data = data.rename(columns={"Datetime": "Date"})
    if "Close" not in data or "Date" not in data:
        raise ValueError(f"Market data for {ticker} must include Date and Close columns.")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date")
    data["Close"] = pd.to_numeric(data["Close"], errors="raise")
    if data.empty:
        raise ValueError(f"No usable closing prices were returned for {ticker}.")
    return data


def get_usd_inr_rate() -> float:
    rates = yf.download(
        "USDINR=X",
        period="5d",
        interval="1d",
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
    )
    if isinstance(rates.columns, pd.MultiIndex):
        rates.columns = [column[0] for column in rates.columns]
    if rates.empty or "Close" not in rates:
        raise ValueError("The latest USD/INR exchange rate is unavailable.")
    close = rates["Close"].dropna().astype(float)
    if close.empty or not np.isfinite(float(close.iloc[-1])):
        raise ValueError("The latest USD/INR exchange rate is unavailable.")
    return float(close.iloc[-1])


@app.get("/health")
def health():
    try:
        _, _, version = load_serving_assets()
        return {"status": "ok", "model_version": version}
    except Exception as exc:
        return {"status": "degraded", "detail": str(exc)}


@app.get("/api/dashboard")
def dashboard_data(ticker: Annotated[str, Query()] = "AAPL"):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported ticker. Choose one of: {', '.join(TICKERS)}.",
        )
    if ticker == "AAPL" and not RAW_CSV.exists():
        raise HTTPException(
            status_code=503,
            detail="Market data is not available. Run python -m src.ingest first.",
        )

    try:
        market_data = load_market_data(ticker)
        usd_inr_rate = get_usd_inr_rate()
    except (OSError, ValueError, pd.errors.ParserError, YFException) as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not load market or exchange-rate data: {exc}"
        ) from exc

    recent = market_data.tail(252)
    latest = recent.iloc[-1]
    ticker_dir = MODELS_DIR / "tickers" / ticker
    metrics_path = METRICS_PATH if ticker == "AAPL" else ticker_dir / "metrics.json"
    history_path = HISTORY_PATH if ticker == "AAPL" else ticker_dir / "history.json"
    model_path = MODEL_PATH if ticker == "AAPL" else ticker_dir / "lstm.keras"
    scaler_path = MODEL_SCALER_PATH if ticker == "AAPL" else ticker_dir / "scaler.joblib"
    model_supported = model_path.exists() and scaler_path.exists()

    metrics = None
    if model_supported and metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not read evaluation metrics: {exc}"
            ) from exc

    history = None
    if model_supported and history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
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
        "ticker": ticker,
        "company": TICKERS[ticker],
        "model_supported": model_supported,
        "usd_inr_rate": usd_inr_rate,
        "observations": int(len(market_data)),
        "latest_date": latest["Date"].date().isoformat(),
        "latest_close": float(latest["Close"]),
        "points": [
            {"date": pd.Timestamp(row.Date).date().isoformat(), "close": float(row.Close)}
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
            "prediction_plot": ticker == "AAPL" and PRED_PLOT_PATH.exists(),
            "loss_plot": ticker == "AAPL" and LOSS_PLOT_PATH.exists(),
            "drift_report": ticker == "AAPL" and DRIFT_REPORT_PATH.exists(),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PriceWindow):
    ticker = payload.ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422, detail=f"Unsupported ticker: {ticker}."
        )
    try:
        model, scaler, version = load_serving_assets(ticker)
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
