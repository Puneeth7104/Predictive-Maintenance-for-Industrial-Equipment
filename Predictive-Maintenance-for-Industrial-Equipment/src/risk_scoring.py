"""Score every machine with the trained model and write the monitoring files."""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import DATA_DIR, HISTORY_DAYS, MODEL_DIR, SENSORS
from .features import build_features


def load_bundle():
    """Load the saved model; retrain automatically if it cannot be loaded (e.g. other scikit-learn version)."""
    path = MODEL_DIR / "model.joblib"
    try:
        bundle = joblib.load(path)
        if bundle.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version differs from the one used for training")
        return bundle
    except Exception as exc:  # noqa: BLE001
        print(f"[score] model not usable ({exc}); retraining ...")
        from .train import train_and_save
        train_and_save()
        return joblib.load(path)


def run():
    bundle = load_bundle()
    model, thr, cols = bundle["model"], bundle["threshold"], bundle["feature_cols"]
    sensors = pd.read_csv(DATA_DIR / "sensor_data.csv", parse_dates=["timestamp"])
    df, _ = build_features(sensors)
    df["risk_score"] = model.predict_proba(df[cols])[:, 1]

    # which sensor is drifting most compared with its normal level?
    z = pd.DataFrame({s: df[f"{s}_dev_baseline"] / df[f"{s}_dev_baseline"].std() for s in SENSORS})
    driver = z.abs().idxmax(axis=1)
    df["main_driver"] = [f"{s} {z.loc[i, s]:+.1f}σ vs baseline" for i, s in driver.items()]

    medium = thr * 0.5
    latest = df.sort_values("timestamp").groupby("machine_id").tail(1).copy()
    latest["risk_level"] = np.where(latest["risk_score"] >= thr, "High",
                                    np.where(latest["risk_score"] >= medium, "Medium", "Low"))
    # a machine with no reading for 3+ hours is stopped (repair/maintenance) -> not alertable
    end_ts = df["timestamp"].max()
    latest.loc[latest["timestamp"] < end_ts - pd.Timedelta(hours=3), "risk_level"] = "Offline"
    keep = ["machine_id", "machine_type", "timestamp", "risk_score", "risk_level", "temperature",
            "vibration", "pressure", "hours_since_maintenance", "main_driver"]
    latest = latest[keep].sort_values("risk_score", ascending=False)
    latest["risk_score"] = latest["risk_score"].round(4)
    latest.to_csv(DATA_DIR / "risk_scores.csv", index=False)

    end = df["timestamp"].max()
    hist = df[df["timestamp"] >= end - pd.Timedelta(days=HISTORY_DAYS)]
    hist = hist[["timestamp", "machine_id", "risk_score", "temperature", "vibration", "pressure"]].copy()
    hist["risk_score"] = hist["risk_score"].round(4)
    hist.to_csv(DATA_DIR / "risk_history.csv", index=False)

    meta = {"threshold": thr, "medium_threshold": medium, "as_of": str(end),
            "model_name": bundle["model_name"], "horizon_h": bundle["horizon_h"]}
    (DATA_DIR / "dashboard_meta.json").write_text(json.dumps(meta, indent=2))
    counts = latest["risk_level"].value_counts().to_dict()
    print(f"[score] as of {end}: {counts}")
    return latest, hist, meta


if __name__ == "__main__":
    run()
