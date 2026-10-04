"""Feature engineering and labelling.

IMPORTANT (no leakage): every feature at time t only uses readings from time t
and earlier (rolling windows, lags, expanding baseline). Future information is
used ONLY to create the label ("does a failure happen in the next H hours?").
"""
import numpy as np
import pandas as pd

from .config import HORIZON_H, SENSORS


def build_features(df):
    """Return (dataframe_with_features, list_of_feature_columns)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    for s in SENSORS:                                   # fill gaps with the last known value (past only)
        df[s] = df.groupby("machine_id")[s].ffill()

    new = {}
    for s in SENSORS:
        g = df.groupby("machine_id")[s]
        for w in (6, 24):
            new[f"{s}_mean_{w}"] = g.transform(lambda x: x.rolling(w, min_periods=3).mean())
            new[f"{s}_std_{w}"] = g.transform(lambda x: x.rolling(w, min_periods=3).std())
        new[f"{s}_max_24"] = g.transform(lambda x: x.rolling(24, min_periods=3).max())
        new[f"{s}_min_24"] = g.transform(lambda x: x.rolling(24, min_periods=3).min())
        new[f"{s}_diff_6"] = df[s] - g.shift(6)         # change over the last 6 h (lag feature)
        new[f"{s}_diff_24"] = df[s] - g.shift(24)       # change over the last 24 h
        baseline = g.transform(lambda x: x.expanding(min_periods=24).mean())
        new[f"{s}_dev_baseline"] = new[f"{s}_mean_6"] - baseline   # drift from the machine's own history

    df = pd.concat([df, pd.DataFrame(new)], axis=1)
    feature_cols = SENSORS + list(new) + ["hours_since_maintenance", "maintenance_count"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols


def add_labels(df, failures, horizon=HORIZON_H):
    """Add next_failure_time, hours_to_failure and the binary label.

    label = 1  if the machine fails within `horizon` hours after this reading.
    """
    df = df.copy()
    nxt_all = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    ts_all = df["timestamp"].values.astype("datetime64[ns]")
    for m, idx in df.groupby("machine_id").groups.items():
        idx = np.asarray(idx)
        ft = np.sort(failures.loc[failures["machine_id"] == m, "timestamp"].values.astype("datetime64[ns]"))
        if len(ft) == 0:
            continue
        pos = np.searchsorted(ft, ts_all[idx], side="right")      # first failure strictly after the reading
        has = pos < len(ft)
        nxt = ft[np.minimum(pos, len(ft) - 1)]
        nxt_all[idx] = np.where(has, nxt, np.datetime64("NaT"))
    df["next_failure_time"] = nxt_all
    df["hours_to_failure"] = (nxt_all - ts_all) / np.timedelta64(1, "h")
    df["label"] = (df["hours_to_failure"] <= horizon).astype(int)
    return df
