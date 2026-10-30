# Stock Price Prediction Pipeline

End-to-end MLOps pipeline that forecasts the **next trading day’s AAPL closing price** from the previous 60 daily closes. The workflow versions data, tracks experiments, registers models, serves predictions, monitors drift, and retrains when new market data arrives.

This is an academic forecasting demonstration. Predictions are **not financial advice**.

## What you get

| Tool | Role |
| --- | --- |
| Python / LSTM | Two-layer Keras LSTM on 60-day Close windows |
| DVC | Raw dataset and pipeline stage versioning |
| MLflow | Experiment tracking and `AAPL_LSTM` model registry |
| FastAPI | `/health` and `/predict` |
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
|   `-- api.py
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
dvc init
dvc remote add -d localstore dvc_storage
dvc add data/raw/aapl.csv
dvc push
```

### 3. Run the DVC pipeline

```powershell
dvc repro
```

This runs preprocess → train → evaluate. Training uses 20 epochs on CPU and usually takes a few minutes.

Equivalent manual commands:

```powershell
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

### 4. Inspect MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000. The champion alias is `AAPL_LSTM@champion`.

### 5. Serve predictions

```powershell
python -m uvicorn src.api:app --reload --port 8000
```

- Dashboard: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Health: `GET /health`
- Predict: `POST /predict` with JSON `{"close": [/* 60 floats, oldest to newest */]}`
- Dashboard data: `GET /api/dashboard` (recent closes, evaluation metrics, pipeline artifacts, and report availability)

Example:

```powershell
python -c "import json,pandas as pd,urllib.request; df=pd.read_csv('data/raw/aapl.csv'); closes=df['Close'].tail(60).tolist(); req=urllib.request.Request('http://127.0.0.1:8000/predict', data=json.dumps({'close': closes}).encode(), headers={'Content-Type':'application/json'}); print(urllib.request.urlopen(req).read())"
```

### 6. Drift and retraining

```powershell
python -m src.retrain
python -m src.retrain --scheduled
```

`--scheduled` trains a candidate even if drift is not detected. The candidate is promoted only when its RMSE on a later held-out period is lower than the current champion.

### 7. Docker

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
3. **Train** — two LSTM layers (50 units), dropout 0.2, Dense(1), Adam, MSE, 20 epochs, `validation_split=0.1`, `shuffle=False`.
4. **Evaluate** — RMSE, MAE, MAPE on the original price scale; plots in `reports/`.
5. **Register** — MLflow run + `AAPL_LSTM` with alias `champion`.
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

The model uses only Close for one ticker. Markets are noisy and driven by information that is not in this dataset. Treat metrics as a pipeline demonstration, not a trading signal.
