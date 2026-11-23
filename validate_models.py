import joblib
import pandas as pd
from tensorflow import keras

from src.api import TICKERS, load_market_data
from src.config import MODELS_DIR, load_params
from src.metrics import inverse_close, regression_metrics
from src.preprocess import prepare_sequences

params = load_params()
results = []

for ticker in TICKERS:
    print(f"\nEvaluating {ticker}...", flush=True)

    data = load_market_data(ticker).set_index("Date")
    prepared = prepare_sequences(data, params)

    model_dir = MODELS_DIR if ticker == "AAPL" else MODELS_DIR / "tickers" / ticker
    model_path = model_dir / "lstm.keras"
    scaler_path = model_dir / "scaler.joblib"

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # LSTM predictions
    pred_scaled = model.predict(prepared["X_test"], verbose=0)
    lstm_pred = inverse_close(scaler, pred_scaled)
    actual = inverse_close(scaler, prepared["y_test"])

    # Naive baseline: tomorrow = today's close
    test_indices = prepared["target_index"][prepared["target_index"] >= prepared["split"]]
    naive_pred = prepared["close"][test_indices - 1].flatten()

    lstm_metrics = regression_metrics(actual, lstm_pred)
    naive_metrics = regression_metrics(actual, naive_pred)

    improvement_rmse = (
        (naive_metrics["rmse"] - lstm_metrics["rmse"])
        / naive_metrics["rmse"] * 100
    )
    improvement_mae = (
        (naive_metrics["mae"] - lstm_metrics["mae"])
        / naive_metrics["mae"] * 100
    )

    results.append({
        "Ticker": ticker,
        "LSTM_RMSE": round(lstm_metrics["rmse"], 4),
        "Naive_RMSE": round(naive_metrics["rmse"], 4),
        "RMSE_Improvement_%": round(improvement_rmse, 2),
        "LSTM_MAE": round(lstm_metrics["mae"], 4),
        "Naive_MAE": round(naive_metrics["mae"], 4),
        "MAE_Improvement_%": round(improvement_mae, 2),
        "LSTM_Better": improvement_rmse > 0 and improvement_mae > 0,
    })

    del model
    keras.backend.clear_session()

df = pd.DataFrame(results).sort_values("RMSE_Improvement_%", ascending=False)

print("\n" + "=" * 85)
print("LSTM vs NAIVE BASELINE — FINAL VALIDATION")
print("=" * 85)
print(df.to_string(index=False))

output = MODELS_DIR / "baseline_comparison.csv"
df.to_csv(output, index=False)
print(f"\nSaved report: {output}")

