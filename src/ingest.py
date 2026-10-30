from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import RAW_CSV, load_params


def fetch_aapl(params: dict | None = None) -> pd.DataFrame:
    settings = (params or load_params())["data"]
    data = yf.download(
        settings["ticker"],
        period=settings["period"],
        interval=settings["interval"],
        auto_adjust=settings["auto_adjust"],
        multi_level_index=False,
        progress=False,
    )
    if data.empty:
        raise ValueError("No AAPL data was returned.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    data = data.sort_index()
    data.index.name = "Date"
    validate_ohlcv(data)
    return data


def validate_ohlcv(data: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required.difference(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    if data.index.duplicated().any():
        raise ValueError("Duplicate dates were found in the dataset.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Dates are not in chronological order.")
    if data[list(required)].isna().any().any():
        raise ValueError("Missing values were found in OHLCV columns.")


def save_raw(data: pd.DataFrame, output: Path = RAW_CSV) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index_label="Date")
    return output


def main() -> None:
    data = fetch_aapl()
    path = save_raw(data)
    print(f"Saved {len(data)} rows to {path}")
    print(f"Date range: {data.index.min().date()} to {data.index.max().date()}")


if __name__ == "__main__":
    main()
