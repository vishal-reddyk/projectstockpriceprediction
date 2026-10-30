from __future__ import annotations

import json

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import DRIFT_REPORT_PATH, REPORTS_DIR


def _walk_for_drift(payload) -> bool | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"drift_detected", "dataset_drift"} and isinstance(value, bool):
                return value
            found = _walk_for_drift(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _walk_for_drift(item)
            if found is not None:
                return found
    return None


def snapshot_to_dict(snapshot) -> dict:
    if hasattr(snapshot, "as_dict"):
        return snapshot.as_dict()
    if hasattr(snapshot, "dict"):
        return snapshot.dict()
    return {}


def generate_drift_report(reference_close: np.ndarray, current_close: np.ndarray) -> bool:
    reference = pd.DataFrame({"Close": np.asarray(reference_close).ravel()})
    current = pd.DataFrame({"Close": np.asarray(current_close).ravel()})
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DRIFT_REPORT_PATH))
    payload = snapshot_to_dict(snapshot)
    drift_json = json.dumps(payload, default=str, indent=2)
    (REPORTS_DIR / "data_drift.json").write_text(drift_json, encoding="utf-8")
    detected = _walk_for_drift(payload)
    if detected is None:
        ref_mean = float(np.mean(reference_close))
        cur_mean = float(np.mean(current_close))
        detected = abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-9) > 0.15
    print(f"Drift report saved to {DRIFT_REPORT_PATH}; drift_detected={detected}")
    return bool(detected)


def main() -> None:
    from src.preprocess import load_raw, prepare_sequences

    prepared = prepare_sequences(load_raw())
    reference = prepared["close"][: prepared["split"]]
    current = prepared["close"][-prepared["window"] :]
    generate_drift_report(reference, current)


if __name__ == "__main__":
    main()
