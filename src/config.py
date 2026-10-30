from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def load_params() -> dict:
    with (ROOT / "params.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tracking_uri() -> str:
    default = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    return os.getenv("MLFLOW_TRACKING_URI", default)


RAW_CSV = ROOT / "data" / "raw" / "aapl.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SEQUENCES_PATH = PROCESSED_DIR / "sequences.npz"
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
MODEL_PATH = MODELS_DIR / "lstm.keras"
MODEL_SCALER_PATH = MODELS_DIR / "scaler.joblib"
HISTORY_PATH = MODELS_DIR / "history.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
DRIFT_REPORT_PATH = REPORTS_DIR / "data_drift.html"
PRED_PLOT_PATH = REPORTS_DIR / "actual_vs_predicted.png"
LOSS_PLOT_PATH = REPORTS_DIR / "loss_curve.png"
