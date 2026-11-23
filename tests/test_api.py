import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import PriceWindow, app, load_serving_assets


def test_dashboard_serves_frontend_and_assets():
    client = TestClient(app)
    page = client.get("/")
    stylesheet = client.get("/static/styles.css")
    assert page.status_code == 200
    assert "Stock price prediction" in page.text
    assert stylesheet.status_code == 200
    assert ".metric-grid" in stylesheet.text


def test_dashboard_returns_recent_market_data(tmp_path, monkeypatch):
    raw_csv = tmp_path / "aapl.csv"
    dates = pd.date_range("2025-01-01", periods=65, freq="B")
    pd.DataFrame({"Date": dates, "Close": range(100, 165)}).to_csv(raw_csv, index=False)
    monkeypatch.setattr("src.api.RAW_CSV", raw_csv)
    monkeypatch.setattr("src.api.METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr("src.api.HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.setattr("src.api.DRIFT_REPORT_PATH", tmp_path / "data_drift.html")
    monkeypatch.setattr("src.api.get_usd_inr_rate", lambda: 83.5)

    response = TestClient(app).get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["company"] == "Apple"
    assert payload["model_supported"] is False
    assert payload["usd_inr_rate"] == 83.5
    assert payload["latest_close"] == 164.0
    assert len(payload["points"]) == 65
    assert len(payload["window"]) == 60
    assert payload["metrics"] is None
    assert payload["pipeline"]["stages"]["monitor"] is False


def test_dashboard_loads_other_company_without_a_trained_model(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=65, freq="B")
    market_data = pd.DataFrame({"Date": dates, "Close": range(100, 165)})
    monkeypatch.setattr("src.api.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.api.load_market_data", lambda ticker: market_data)
    monkeypatch.setattr("src.api.get_usd_inr_rate", lambda: 83.5)

    response = TestClient(app).get("/api/dashboard?ticker=NVDA")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "NVDA"
    assert payload["company"] == "NVIDIA"
    assert payload["model_supported"] is False
    assert payload["latest_close"] == 164.0
    assert len(payload["window"]) == 60
    assert payload["metrics"] is None


def test_dashboard_loads_metrics_for_a_trained_ticker(tmp_path, monkeypatch):
    dates = pd.date_range("2025-01-01", periods=65, freq="B")
    market_data = pd.DataFrame({"Date": dates, "Close": range(100, 165)})
    ticker_dir = tmp_path / "tickers" / "NVDA"
    ticker_dir.mkdir(parents=True)
    (ticker_dir / "lstm.keras").touch()
    (ticker_dir / "scaler.joblib").touch()
    (ticker_dir / "metrics.json").write_text(
        '{"rmse": 1.5, "mae": 1.0, "mape_percent": 1.2, "test_sequences": 10}',
        encoding="utf-8",
    )
    (ticker_dir / "history.json").write_text('{"loss": [0.2, 0.1]}', encoding="utf-8")
    monkeypatch.setattr("src.api.MODELS_DIR", tmp_path)
    monkeypatch.setattr("src.api.load_market_data", lambda ticker: market_data)
    monkeypatch.setattr("src.api.get_usd_inr_rate", lambda: 83.5)

    response = TestClient(app).get("/api/dashboard?ticker=NVDA")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_supported"] is True
    assert payload["metrics"]["rmse"] == 1.5
    assert payload["history"]["loss"] == [0.2, 0.1]
    assert len(payload["window"]) == 60


def test_dashboard_rejects_unsupported_ticker():
    response = TestClient(app).get("/api/dashboard?ticker=UNKNOWN")
    assert response.status_code == 422


def test_price_window_requires_sixty_values():
    with __import__("pytest").raises(Exception):
        PriceWindow(close=[1.0] * 59)


def test_health_without_model():
    load_serving_assets.cache_clear()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_predict_rejects_wrong_length():
    client = TestClient(app)
    response = client.post("/predict", json={"close": [1.0] * 10})
    assert response.status_code == 422


def test_predict_reports_ticker_without_a_trained_model(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.MODELS_DIR", tmp_path)
    client = TestClient(app)
    response = client.post("/predict", json={"ticker": "NVDA", "close": [1.0] * 60})
    assert response.status_code == 503
    assert "No trained prediction model" in response.json()["detail"]


def test_predict_uses_selected_ticker_model(monkeypatch):
    class Scaler:
        def transform(self, values):
            return values

        def inverse_transform(self, values):
            return np.asarray(values)

    class Model:
        def predict(self, values, verbose=0):
            assert values.shape == (1, 60, 1)
            return np.asarray([[123.0]])

    monkeypatch.setattr(
        "src.api.load_serving_assets", lambda ticker: (Model(), Scaler(), "7")
    )

    response = TestClient(app).post(
        "/predict", json={"ticker": "NVDA", "close": [100.0] * 60}
    )

    assert response.status_code == 200
    assert response.json() == {"predicted_close": 123.0, "model_version": "7"}
