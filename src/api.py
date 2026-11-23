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
from sklearn.preprocessing import MinMaxScaler
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

# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class PriceWindow(BaseModel):
    close: list[float] = Field(min_length=60, max_length=60)
    ticker: str = "AAPL"


class TrainRequest(BaseModel):
    ticker: str


class PredictionResponse(BaseModel):
    predicted_close: float
    model_version: str


# =========================================================
# TICKER CONFIGURATION
# =========================================================

TICKERS = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
}


# =========================================================
# MODEL LOADING
# =========================================================

@lru_cache(maxsize=16)
def load_serving_assets(ticker: str = "AAPL"):
    ticker = ticker.upper()

    # First: load a ticker-specific trained model if available
    ticker_dir = MODELS_DIR / "tickers" / ticker
    model_path = ticker_dir / "lstm.keras"
    scaler_path = ticker_dir / "scaler.joblib"

    if model_path.exists() and scaler_path.exists():
        from tensorflow import keras

        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        metrics_path = ticker_dir / "metrics.json"
        version = "local"

        if metrics_path.exists():
            metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            version = str(metrics.get("model_version", version))

        return model, scaler, version

    # Second: preserve the original AAPL MLflow champion fallback
    if ticker == "AAPL":
        params = load_params()
        mlflow.set_tracking_uri(tracking_uri())
        client = mlflow.MlflowClient()

        name = params["mlflow"]["registered_model"]
        alias = params["mlflow"]["alias"]

        try:
            version = client.get_model_version_by_alias(name, alias)

            model = mlflow.keras.load_model(
                f"models:/{name}/{version.version}"
            )

            scaler_path = mlflow.artifacts.download_artifacts(
                run_id=version.run_id,
                artifact_path="scaler.joblib",
            )

            scaler = joblib.load(scaler_path)

            return model, scaler, str(version.version)

        except Exception:
            # Original local AAPL fallback
            if MODEL_PATH.exists() and MODEL_SCALER_PATH.exists():
                from tensorflow import keras

                model = keras.models.load_model(MODEL_PATH)
                scaler = joblib.load(MODEL_SCALER_PATH)

                return model, scaler, "local"

            raise RuntimeError(
                "No trained AAPL ticker model, registered MLflow champion, "
                "or local model artifacts were found."
            )

    raise RuntimeError(
        f"No trained prediction model is available for {ticker}."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="StockPulse ML Stock Prediction API",
    description="Stock market dashboard, LSTM training, and prediction API.",
    version="1.0.0",
)


# =========================================================
# MARKET DATA
# =========================================================

def load_market_data(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()

    if ticker == "AAPL" and RAW_CSV.exists():
        data = pd.read_csv(RAW_CSV, parse_dates=["Date"])

        if "Date" not in data or "Close" not in data:
            raise ValueError(
                "The AAPL CSV must include Date and Close columns."
            )

        data["Date"] = pd.to_datetime(
            data["Date"], errors="raise"
            )

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
            raise ValueError(
                f"No market data was returned for {ticker}."
            )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [
                column[0] for column in data.columns
            ]

        data = data.reset_index()

        if "Date" not in data and "Datetime" in data:
            data = data.rename(
                columns={"Datetime": "Date"}
            )

    if "Close" not in data or "Date" not in data:
        raise ValueError(
            f"Market data for {ticker} must include Date and Close columns."
        )

    data = data.dropna(
        subset=["Date", "Close"]
    ).sort_values("Date")

    data["Close"] = pd.to_numeric(
        data["Close"], errors="raise"
    )

    if data.empty:
        raise ValueError(
            f"No usable closing prices were returned for {ticker}."
        )

    return data


# =========================================================
# USD / INR EXCHANGE RATE
# =========================================================

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
        rates.columns = [
            column[0] for column in rates.columns
        ]

    if rates.empty or "Close" not in rates:
        raise ValueError(
            "The latest USD/INR exchange rate is unavailable."
        )

    close = rates["Close"].dropna().astype(float)

    if close.empty or not np.isfinite(float(close.iloc[-1])):
        raise ValueError(
            "The latest USD/INR exchange rate is unavailable."
        )

    return float(close.iloc[-1])


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    try:
        _, _, version = load_serving_assets("AAPL")

        return {
            "status": "ok",
            "model_version": version,
        }

    except Exception as exc:
        return {
            "status": "degraded",
            "detail": str(exc),
        }


# =========================================================
# DASHBOARD API
# =========================================================

@app.get("/api/dashboard")
def dashboard_data(
    ticker: Annotated[str, Query()] = "AAPL"
):
    ticker = ticker.upper()

    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported ticker. Choose one of: "
                f"{', '.join(TICKERS)}."
            ),
        )


    try:
        market_data = load_market_data(ticker)
        usd_inr_rate = get_usd_inr_rate()

    except (
        OSError,
        ValueError,
        pd.errors.ParserError,
        YFException,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not load market or exchange-rate data: {exc}",
        ) from exc

    recent = market_data.tail(252)
    latest = recent.iloc[-1]

    # Ticker-specific model artifacts
    ticker_dir = MODELS_DIR / "tickers" / ticker

    ticker_model_path = ticker_dir / "lstm.keras"
    ticker_scaler_path = ticker_dir / "scaler.joblib"
    ticker_metrics_path = ticker_dir / "metrics.json"
    ticker_history_path = ticker_dir / "history.json"

    ticker_model_supported = (
        ticker_model_path.exists()
        and ticker_scaler_path.exists()
    )

    # AAPL can still use its original MLflow/local model
    if ticker == "AAPL" and not ticker_model_supported:
        model_path = MODEL_PATH
        scaler_path = MODEL_SCALER_PATH
        metrics_path = METRICS_PATH
        history_path = HISTORY_PATH

    else:
        model_path = ticker_model_path
        scaler_path = ticker_scaler_path
        metrics_path = ticker_metrics_path
        history_path = ticker_history_path

    model_supported = (
        model_path.exists()
        and scaler_path.exists()
    )

    # Metrics
    metrics = None

    if model_supported and metrics_path.exists():
        try:
            metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )

        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read evaluation metrics: {exc}",
            ) from exc

    # Training history
    history = None

    if model_supported and history_path.exists():
        try:
            history = json.loads(
                history_path.read_text(encoding="utf-8")
            )

        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read model training history: {exc}",
            ) from exc

    # Existing project pipeline stages
    stages = {
        "ingest": RAW_CSV.exists(),
        "preprocess": (
            ROOT / "data" / "processed" / "sequences.npz"
        ).exists(),
        "train": (
            MODEL_PATH.exists()
            and MODEL_SCALER_PATH.exists()
        ),
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
            {
                "date": pd.Timestamp(row.Date).date().isoformat(),
                "close": float(row.Close),
            }
            for row in recent.itertuples(index=False)
        ],

        "window": [
            float(value)
            for value in market_data["Close"].tail(60)
        ],

        "metrics": metrics,
        "history": history,

        "pipeline": {
            "dvc_initialized": (
                ROOT / ".dvc" / "config"
            ).exists(),
            "stages": stages,
        },

        "reports": {
            "prediction_plot": (
                ticker == "AAPL"
                and PRED_PLOT_PATH.exists()
            ),
            "loss_plot": (
                ticker == "AAPL"
                and LOSS_PLOT_PATH.exists()
            ),
            "drift_report": (
                ticker == "AAPL"
                and DRIFT_REPORT_PATH.exists()
            ),
        },
    }


# =========================================================
# PREDICTION API
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(payload: PriceWindow):
    ticker = payload.ticker.upper()

    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported ticker: {ticker}.",
        )

    try:
        model, scaler, version = load_serving_assets(ticker)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    window = np.array(
        payload.close,
        dtype=float,
    ).reshape(-1, 1)

    scaled = scaler.transform(window)

    scaled_prediction = model.predict(
        scaled.reshape(1, 60, 1),
        verbose=0,
    )

    price = scaler.inverse_transform(
        scaled_prediction
    )[0, 0]

    return PredictionResponse(
        predicted_close=float(price),
        model_version=version,
    )


# =========================================================
# MODEL TRAINING API
# =========================================================

@app.post("/train")
def train_model(payload: TrainRequest):
    ticker = payload.ticker.upper()

    # Validate ticker
    if ticker not in TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported ticker: {ticker}",
        )

    ticker_dir = MODELS_DIR / "tickers" / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -------------------------------------------------
        # 1. Download historical stock data
        # -------------------------------------------------
        data = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            raise ValueError(
                "No historical stock data was returned."
            )

        # Handle yfinance multi-index columns
        close = data["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        values = (
            close.dropna()
            .values
            .reshape(-1, 1)
            .astype(float)
        )

        if len(values) < 200:
            raise ValueError(
                "Insufficient historical data to train the model."
            )

        # -------------------------------------------------
        # 2. Chronological dataset split
        # -------------------------------------------------
        n = len(values)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        # -------------------------------------------------
        # 3. Scale data using training data only
        # -------------------------------------------------
        scaler = MinMaxScaler()

        scaler.fit(values[:train_end])

        scaled_values = scaler.transform(values)

        # -------------------------------------------------
        # 4. Prepare 60-day sequences
        # -------------------------------------------------
        X = []
        y = []
        target_indices = []

        for i in range(60, len(scaled_values)):
            X.append(
                scaled_values[i - 60:i]
            )
            y.append(
                scaled_values[i]
            )
            target_indices.append(i)

        X = np.array(X)
        y = np.array(y)
        target_indices = np.array(target_indices)

        # -------------------------------------------------
        # 5. Split sequences chronologically
        # -------------------------------------------------
        train_mask = target_indices < train_end

        val_mask = (
            (target_indices >= train_end)
            & (target_indices < val_end)
        )

        test_mask = target_indices >= val_end

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_val = X[val_mask]
        y_val = y[val_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        if (
            len(X_train) == 0
            or len(X_val) == 0
            or len(X_test) == 0
        ):
            raise ValueError(
                "Train, validation, or test dataset is empty."
            )

        # -------------------------------------------------
        # 6. Build LSTM model
        # -------------------------------------------------
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(
                shape=(60, 1)
            ),
            keras.layers.LSTM(32),
            keras.layers.Dense(
                16,
                activation="relu"
            ),
            keras.layers.Dense(1),
        ])

        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=["mae"],
        )

        # -------------------------------------------------
        # 7. Train the model
        # -------------------------------------------------
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        # -------------------------------------------------
        # 8. Test the model
        # -------------------------------------------------
        predictions = model.predict(
            X_test,
            verbose=0,
        )

        predictions_actual = scaler.inverse_transform(
            predictions
        )

        actual_values = scaler.inverse_transform(
            y_test
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (predictions_actual - actual_values) ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    predictions_actual - actual_values
                )
            )
        )
        mape = float(
    np.mean(
        np.abs((actual_values - predictions_actual) / actual_values)
    ) * 100
)

        # -------------------------------------------------
        # 9. Save model and scaler
        # -------------------------------------------------
        model_path = ticker_dir / "lstm.keras"
        scaler_path = ticker_dir / "scaler.joblib"

        model.save(model_path)
        joblib.dump(
            scaler,
            scaler_path,
        )

        # -------------------------------------------------
        # 10. Save evaluation metrics
        # -------------------------------------------------
        metrics = {
            "ticker": ticker,
            "company": TICKERS[ticker],
            "rmse": rmse,
            "mae": mae,
            "mape_percent": mape,
            "epochs": len(history.history["loss"]),
            "train_sequences": int(len(X_train)),
            "validation_sequences": int(len(X_val)),
            "test_sequences": int(len(X_test)),
            "model_version": "1",
        }

        (ticker_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        # -------------------------------------------------
        # 11. Save training history
        # -------------------------------------------------
        history_data = {
            "loss": [
                float(value)
                for value in history.history["loss"]
            ],
            "val_loss": [
                float(value)
                for value in history.history.get(
                    "val_loss", []
                )
            ],
        }

        (ticker_dir / "history.json").write_text(
            json.dumps(history_data, indent=2),
            encoding="utf-8",
        )

        # -------------------------------------------------
        # 12. Clear model cache
        # -------------------------------------------------
        load_serving_assets.cache_clear()

        return {
            "status": "success",
            "ticker": ticker,
            "company": TICKERS[ticker],
            "metrics": metrics,
            "message": (
                f"{ticker} model trained successfully."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed for {ticker}: {str(exc)}",
        ) from exc


# =========================================================
# STATIC FILES AND REPORTS
# =========================================================

app.mount(
    "/reports",
    StaticFiles(
        directory=REPORTS_DIR,
        check_dir=False,
    ),
    name="reports",
)

app.mount(
    "/static",
    StaticFiles(
        directory=ROOT / "src" / "static",
        check_dir=False,
    ),
    name="static",
)


# =========================================================
# FRONTEND
# =========================================================

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(
        ROOT / "src" / "static" / "index.html"
    )
