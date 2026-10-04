"""Train, compare and evaluate failure-prediction models."""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import plots
from .config import DATA_DIR, HORIZON_H, MODEL_DIR, RECALL_TARGET, SEED
from .features import add_labels, build_features


def load_labelled():
    sensors = pd.read_csv(DATA_DIR / "sensor_data.csv", parse_dates=["timestamp"])
    log = pd.read_csv(DATA_DIR / "maintenance_log.csv", parse_dates=["timestamp"])
    df, cols = build_features(sensors)
    df = add_labels(df, log[log["event"] == "failure"])
    return df, cols, log


def time_split(df, horizon=HORIZON_H):
    """Chronological 60/15/25 split with a `horizon`-hour gap so labels never overlap."""
    t = df["timestamp"]
    span = t.max() - t.min()
    b1, b2 = t.min() + span * 0.60, t.min() + span * 0.75
    gap = pd.Timedelta(hours=horizon)
    return (df[t < b1], df[(t >= b1 + gap) & (t < b2)], df[t >= b2 + gap])


def model_zoo():
    return {
        "Logistic (no weights)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Logistic (balanced)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest (balanced)": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                                           class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "Random Forest (undersampled 5:1)": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                                                   n_jobs=-1, random_state=SEED),
        "Gradient Boosting (balanced)": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=250,
                                                                       class_weight="balanced", random_state=SEED),
    }


def undersample(X, y, ratio=5, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep = rng.choice(neg, size=min(len(neg), ratio * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, keep]))
    return X.iloc[idx], y.iloc[idx]


def choose_threshold(y, score, recall_target=RECALL_TARGET):
    """Highest-precision threshold that still reaches the recall target (else best F2)."""
    p, r, thr = precision_recall_curve(y, score)
    p, r = p[:-1], r[:-1]
    ok = r >= recall_target
    if ok.any():
        return float(thr[ok][np.argmax(p[ok])])
    f2 = 5 * p * r / np.maximum(4 * p + r, 1e-9)
    return float(thr[np.argmax(f2)])


def event_metrics(d, pred, n_machines):
    """Alert-level view: failures caught, lead time, false-alarm episodes per machine-week."""
    d = d.assign(pred=pred).sort_values(["machine_id", "timestamp"])
    pos = d[d["label"] == 1]
    ev = pos.groupby(["machine_id", "next_failure_time"])
    caught, leads = 0, []
    for _, g in ev:
        hit = g[g["pred"] == 1]
        if len(hit):
            caught += 1
            leads.append(float(hit["hours_to_failure"].max()))
    n_events = ev.ngroups

    a = d[d["pred"] == 1].copy()
    false_eps = 0
    if len(a):
        gap = a.groupby("machine_id")["timestamp"].diff() > pd.Timedelta(hours=6)
        a["episode"] = (gap | a["machine_id"].ne(a["machine_id"].shift())).cumsum()
        eps = a.groupby("episode")["label"].max()
        false_eps = int((eps == 0).sum())
        n_eps = int(len(eps))
    else:
        n_eps = 0
    weeks = (d["timestamp"].max() - d["timestamp"].min()).total_seconds() / (7 * 86400)
    return {
        "failures_in_test": int(n_events),
        "failures_caught": int(caught),
        "failure_catch_rate": float(caught / n_events) if n_events else None,
        "mean_lead_time_h": float(np.mean(leads)) if leads else None,
        "alert_episodes": n_eps,
        "false_alarm_episodes": false_eps,
        "false_alarms_per_machine_week": float(false_eps / (n_machines * weeks)),
    }


def row_metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def train_and_save():
    df, cols, _ = load_labelled()
    train, val, test = time_split(df)
    Xtr, ytr = train[cols], train["label"]
    Xva, yva = val[cols], val["label"]
    Xte, yte = test[cols], test["label"]
    print(f"[train] rows train/val/test = {len(train):,}/{len(val):,}/{len(test):,}; "
          f"positive rate = {ytr.mean():.2%}/{yva.mean():.2%}/{yte.mean():.2%}")

    plots.sensor_profile(df)
    fitted, comparison = {}, []
    for name, model in model_zoo().items():
        X_fit, y_fit = undersample(Xtr, ytr) if "undersampled" in name else (Xtr, ytr)
        model.fit(X_fit, y_fit)
        fitted[name] = model
        sv = model.predict_proba(Xva)[:, 1]
        thr = choose_threshold(yva, sv)
        st = model.predict_proba(Xte)[:, 1]
        comparison.append({"model": name, "val_pr_auc": float(average_precision_score(yva, sv)),
                           **{f"test_{k}": v for k, v in row_metrics(yte, st, thr).items()}})
        print(f"[train] {name:34s} val PR-AUC={comparison[-1]['val_pr_auc']:.3f} "
              f"test PR-AUC={comparison[-1]['test_pr_auc']:.3f}")

    best = max(comparison, key=lambda c: c["val_pr_auc"])["model"]
    model = fitted[best]
    sv = model.predict_proba(Xva)[:, 1]
    thr = choose_threshold(yva, sv)
    st = model.predict_proba(Xte)[:, 1]
    pred = (st >= thr).astype(int)
    test = test.assign(score=st)

    selected = row_metrics(yte, st, thr)
    cm = confusion_matrix(yte, pred)
    selected["confusion_matrix"] = cm.tolist()
    selected.update(event_metrics(test, pred, df["machine_id"].nunique()))

    imp = permutation_importance(model, Xva, yva, scoring="average_precision", n_repeats=3,
                                 random_state=SEED, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1][:12]
    top_features = [{"feature": cols[i], "importance": float(imp.importances_mean[i])} for i in order]

    plots.pr_curves(yte, {n: m.predict_proba(Xte)[:, 1] for n, m in fitted.items()})
    plots.confusion(cm)
    plots.importance(cols, imp.importances_mean)
    plots.model_comparison([c["model"] for c in comparison], [c["val_pr_auc"] for c in comparison])
    fails = test[(test["label"] == 1)].drop_duplicates(["machine_id", "next_failure_time"])
    if len(fails):
        row = fails.iloc[len(fails) // 2]
        plots.timeline(test, row["machine_id"], row["next_failure_time"], thr)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "feature_cols": cols, "model_name": best,
                 "horizon_h": HORIZON_H, "sklearn_version": sklearn.__version__}, MODEL_DIR / "model.joblib")
    metrics = {
        "model_name": best, "horizon_h": HORIZON_H, "recall_target": RECALL_TARGET,
        "dataset": {"rows_after_features": int(len(df)), "machines": int(df["machine_id"].nunique()),
                    "failures": int(df.drop_duplicates(["machine_id", "next_failure_time"])["next_failure_time"].notna().sum()),
                    "positive_rate": float(df["label"].mean()), "n_features": len(cols),
                    "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
                    "train_end": str(train["timestamp"].max()), "test_start": str(test["timestamp"].min()),
                    "test_end": str(test["timestamp"].max())},
        "selected_test": selected, "comparison": comparison, "top_features": top_features,
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] selected: {best}; threshold={thr:.3f}; "
          f"test precision={selected['precision']:.3f} recall={selected['recall']:.3f} "
          f"PR-AUC={selected['pr_auc']:.3f}; caught {selected['failures_caught']}/{selected['failures_in_test']} failures")
    return metrics


if __name__ == "__main__":
    train_and_save()
    sys.exit(0)
