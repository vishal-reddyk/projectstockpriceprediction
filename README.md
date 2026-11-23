# Stock Price Prediction Pipeline

End-to-end MLOps pipeline and dashboard for exploring AAPL, NVDA, MSFT, AMZN, META, GOOGL, and TSLA. Each ticker has its own LSTM model trained on the previous 60 daily closes. The dashboard displays historical prices, per-ticker evaluation metrics, and next-trading-day estimates, in USD or converted to INR using the latest available USD/INR market rate. The workflow versions the primary AAPL pipeline with DVC, tracks experiments and registers models with MLflow, serves predictions with FastAPI, and monitors drift with Evidently.

This is an academic forecasting demonstration. Predictions are **not financial advice**.

## What you get

| Tool | Role |
| --- | --- |
| Python / LSTM | Separate two-layer Keras LSTM models on 60-day Close windows |
| DVC | Raw dataset and pipeline stage versioning |
| MLflow | Per-ticker experiments and registered model versions |
| FastAPI | Dashboard, `/health`, `/api/dashboard`, and `/predict` |
| Docker | Prediction service image `aapl-prediction` |
| GitHub Actions | Lint, tests, image build, scheduled retrain |
| Evidently | HTML data-drift report vs the last 60 closes |

## Project layout

```text
stock-price-prediction-pipeline/
|-- .github/workflows/
|   |-- ci.yml
|   `-- retrain.yml
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- reports/
|-- src/
|   |-- ingest.py
|   |-- preprocess.py
|   |-- train.py
|   |-- evaluate.py
|   |-- retrain.py
|   |-- train_ticker.py
|   `-- api.py
|-- src/static/
|   |-- index.html
|   |-- dashboard.js
|   `-- styles.css
|-- tests/
|-- dvc.yaml
|-- Dockerfile
`-- requirements.txt
```

## Requirements

- Python 3.11 or 3.12 (TensorFlow does not support every newer Python on Windows; 3.13+ often fails)
- Git
- Optional: Docker Desktop, DVC, a GitHub self-hosted runner labeled `aapl-mlops`

## Quick start

```powershell
cd C:\Users\karri\Desktop\stock-price-prediction-pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `py -3.11` is not installed, install Python 3.11 and retry. Python 3.13 often cannot install TensorFlow.

### 1. Ingest five years of AAPL data

```powershell
python -m src.ingest
```

### 2. Version the raw CSV with DVC

```powershell
dvc add data/raw/aapl.csv
dvc push
```

The repository is already initialized for DVC and uses `dvc_storage` as its local remote.

### 3. Run the AAPL DVC pipeline

```powershell
dvc repro
```

This runs preprocess → train → evaluate for AAPL. Training uses the settings in `params.yaml` (currently 5 epochs).

Equivalent manual commands:

```powershell
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

### 4. Train separate models for the other tickers

```powershell
python -m src.train_ticker
```

By default this downloads five years of daily data for configured tickers that do not already have a saved model, then trains, evaluates, and registers each model separately. Train just one ticker with `python -m src.train_ticker --ticker NVDA`, or deliberately retrain all configured tickers with `python -m src.train_ticker --all`. Artifacts are saved under `models/tickers/<TICKER>/`; metrics and model versions are logged to ticker-specific MLflow experiments and registered models (for example, `NVDA_LSTM`).

### 5. Inspect MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000. Each model is registered with a `champion` alias, for example `AAPL_LSTM@champion` or `NVDA_LSTM@champion`.

### 6. Serve predictions

```powershell
python -m uvicorn src.api:app --reload --port 8000
```

- Dashboard: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Health: `GET /health`
- Select AAPL, NVDA, MSFT, AMZN, META, GOOGL, or TSLA in the dashboard to view its market history; select USD or INR to change the displayed currency.
- Select AAPL, NVDA, MSFT, AMZN, META, GOOGL, or TSLA in the dashboard to view its market history and trained-model evaluation; select USD or INR to change the displayed currency.
- Predict: `POST /predict` with JSON `{"ticker": "NVDA", "close": [/* 60 floats, oldest to newest */]}`. The ticker must have a trained local model.
- Dashboard data: `GET /api/dashboard?ticker=NVDA` returns recent closes, model availability, metrics when trained, and the latest USD/INR rate.

Example:

```powershell
python -c "import json,pandas as pd,urllib.request; df=pd.read_csv('data/raw/aapl.csv'); closes=df['Close'].tail(60).tolist(); req=urllib.request.Request('http://127.0.0.1:8000/predict', data=json.dumps({'close': closes}).encode(), headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req).read())"
```

### 7. Drift and retraining

```powershell
python -m src.retrain
python -m src.retrain --scheduled
```

`--scheduled` trains a candidate even if drift is not detected. The candidate is promoted only when its RMSE on a later held-out period is lower than the current champion.

### 8. Docker

```powershell
docker build -t aapl-prediction .
docker run --rm -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 aapl-prediction
```

Point `MLFLOW_TRACKING_URI` at a tracking server the container can reach. A host-only SQLite file is not visible inside the container unless you mount it or run MLflow as a server.

To run the dashboard and API with the local pipeline outputs mounted into the container:

```powershell
docker compose up --build
```

Open http://127.0.0.1:8000. Run the pipeline first to populate `data/`, `models/`, and `reports/`; the dashboard reports unavailable stages when those artifacts have not been generated yet.

## Pipeline details

1. **Ingest** — `yfinance` download of AAPL OHLCV with `auto_adjust=False`, then empty/missing/duplicate/order checks.
2. **Preprocess** — chronological 80/20 split, `MinMaxScaler` fit on training Close only, 60-day windows, next-day Close target.
3. **Train** — two LSTM layers (50 units), dropout 0.2, Dense(1), Adam, MSE, 5 epochs by default, `validation_split=0.1`, `shuffle=False`. The AAPL DVC pipeline uses `src.train`; other ticker-specific models use `src.train_ticker`.
4. **Evaluate** — RMSE, MAE, MAPE on the original price scale; plots in `reports/`.
5. **Register** — a separate MLflow experiment and model per ticker (for example, `AAPL_LSTM` or `NVDA_LSTM`) with alias `champion`.
6. **Monitor** — Evidently compares training Close vs the latest 60 closes.
7. **Retrain** — fetch new data, optional train, RMSE gate before promotion.

Hyperparameters live in `params.yaml`.

## Tests and lint

```powershell
ruff check .
pytest
```

## GitHub Actions

- `ci.yml` runs on push: install, `ruff`, `pytest`, `docker build`.
- `retrain.yml` runs weekly (Sunday 00:00 UTC) or manually on a self-hosted runner with the `aapl-mlops` label so it can reach local DVC storage and the MLflow registry.

## Limitations

Each LSTM uses only the Close series for its own ticker. These are independent experimental models; their outputs do not use news, fundamentals, market-wide features, or transaction costs and should not be used as trading signals. The DVC reproduction pipeline covers AAPL; ticker-specific training is currently run with `src.train_ticker`. INR amounts are currency conversions of USD-denominated prices, not prices from an Indian exchange. Markets are noisy. Predictions are not financial advice.
