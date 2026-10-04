import numpy as np
import pandas as pd

from src import alerts
from src.features import add_labels, build_features
from src.generate_data import generate


def small():
    return generate(n_machines=3, n_days=40, seed=1)


def test_features_do_not_use_future_data():
    sensors, _ = small()
    full, cols = build_features(sensors)
    cut_time = sensors["timestamp"].quantile(0.6)
    part, _ = build_features(sensors[sensors["timestamp"] <= cut_time])
    merged = part.merge(full, on=["machine_id", "timestamp"], suffixes=("_p", "_f"))
    assert len(merged) == len(part) > 0
    for c in cols:
        assert np.allclose(merged[f"{c}_p"], merged[f"{c}_f"], equal_nan=True), c


def test_label_marks_only_the_hours_before_failure():
    sensors, log = small()
    df, _ = build_features(sensors)
    df = add_labels(df, log[log["event"] == "failure"], horizon=24)
    pos = df[df["label"] == 1]
    assert 0 < len(pos) < len(df)
    assert (pos["hours_to_failure"] <= 24).all() and (pos["hours_to_failure"] > 0).all()
    assert (df.loc[df["label"] == 0, "hours_to_failure"].fillna(1e9) > 24).all()


def test_alert_message_lists_only_high_risk_and_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "DATA_DIR", tmp_path)
    for k in ("SLACK_WEBHOOK_URL", "SMTP_HOST", "ALERT_TO"):
        monkeypatch.delenv(k, raising=False)
    scores = pd.DataFrame([
        {"machine_id": "M01", "machine_type": "Pump", "risk_score": 0.99, "risk_level": "High",
         "main_driver": "vibration +5.0σ vs baseline", "hours_since_maintenance": 500},
        {"machine_id": "M02", "machine_type": "Motor", "risk_score": 0.01, "risk_level": "Low",
         "main_driver": "pressure +0.1σ vs baseline", "hours_since_maintenance": 100}])
    res = alerts.dispatch(scores)
    assert res["dry_run"] and res["high_risk_machines"] == 1
    assert "M01" in res["message"] and "M02" not in res["message"]
    assert (tmp_path / "alerts_log.csv").exists()
