"""Figures used in the report and the dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from .config import FIG_DIR, HORIZON_H

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def sensor_profile(df):
    """Average sensor drift in the hours before a failure."""
    d = df[(df["hours_to_failure"] <= 120)].copy()
    d["h"] = d["hours_to_failure"].round()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, s in zip(axes, ["temperature", "vibration", "pressure"]):
        g = d.groupby("h")[f"{s}_dev_baseline"].mean()
        ax.plot(g.index, g.values, color="#c0392b")
        ax.axvline(HORIZON_H, ls="--", color="grey")
        ax.invert_xaxis()
        ax.set_title(f"{s}: drift from baseline")
        ax.set_xlabel("hours before failure")
    _save(fig, "sensor_profile.png")


def pr_curves(y, scores_by_model, name="pr_curves.png"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model, sc in scores_by_model.items():
        p, r, _ = precision_recall_curve(y, sc)
        ax.plot(r, p, label=model)
    ax.axhline(y.mean(), ls=":", color="grey", label="no-skill")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test set)")
    ax.legend(fontsize=7)
    _save(fig, name)


def confusion(cm, name="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Normal", "At risk"])
    ax.set_yticks([0, 1], ["Normal", "At risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test)")
    _save(fig, name)


def importance(names, values, name="feature_importance.png"):
    order = np.argsort(values)[-12:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(np.array(names)[order], np.array(values)[order], color="#2c7fb8")
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    ax.set_title("Top features (permutation importance)")
    _save(fig, name)


def model_comparison(names, pr_aucs, name="model_comparison.png"):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, pr_aucs, color="#41ab5d")
    ax.set_xlabel("PR-AUC (validation)")
    ax.set_title("Model comparison")
    _save(fig, name)


def timeline(test, machine, fail_time, threshold, name="risk_timeline.png"):
    w = test[(test["machine_id"] == machine) &
             (test["timestamp"] >= fail_time - np.timedelta64(5 * 24, "h")) &
             (test["timestamp"] <= fail_time)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.plot(w["timestamp"], w["score"], color="#c0392b")
    a1.axhline(threshold, ls="--", color="grey", label="alert threshold")
    a1.axvline(fail_time, color="black", label="failure")
    a1.set_ylabel("risk score")
    a1.set_title(f"Machine {machine}: risk score before a failure")
    a1.legend(fontsize=8)
    a2.plot(w["timestamp"], w["temperature"], color="#e67e22", label="temperature")
    a2.set_ylabel("temperature")
    b = a2.twinx()
    b.plot(w["timestamp"], w["vibration"], color="#2c3e50", label="vibration")
    b.set_ylabel("vibration")
    a2.axvline(fail_time, color="black")
    fig.autofmt_xdate()
    _save(fig, name)
