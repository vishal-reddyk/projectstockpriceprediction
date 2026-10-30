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

    response = TestClient(app).get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["latest_close"] == 164.0
    assert len(payload["points"]) == 65
    assert len(payload["window"]) == 60
    assert payload["metrics"] is None
    assert payload["pipeline"]["stages"]["monitor"] is False


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
