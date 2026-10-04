"""Streamlit monitoring app:  streamlit run app.py"""
import json
import os

import pandas as pd
import streamlit as st

from src import alerts
from src.config import DATA_DIR, FIG_DIR, MODEL_DIR

st.set_page_config(page_title="Predictive Maintenance", page_icon="🛠️", layout="wide")

# make Streamlit secrets available to the alert module (Slack / e-mail)
for key in ("SLACK_WEBHOOK_URL", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_FROM", "ALERT_TO"):
    try:
        if key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
    except Exception:  # no secrets file -> alerts stay in dry-run mode
        pass

REQUIRED = [DATA_DIR / "risk_scores.csv", DATA_DIR / "risk_history.csv",
            DATA_DIR / "dashboard_meta.json", MODEL_DIR / "metrics.json"]


@st.cache_data(show_spinner="Preparing data and model (first start only)...")
def load():
    if not all(p.exists() for p in REQUIRED):
        from src.pipeline import main
        main()
    scores = pd.read_csv(DATA_DIR / "risk_scores.csv", parse_dates=["timestamp"])
    hist = pd.read_csv(DATA_DIR / "risk_history.csv", parse_dates=["timestamp"])
    meta = json.loads((DATA_DIR / "dashboard_meta.json").read_text())
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    return scores, hist, meta, metrics


scores, hist, meta, metrics = load()
ICON = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low", "Offline": "⚪ Stopped"}

st.title("🛠️ Predictive Maintenance for Industrial Equipment")
st.caption(f"Model: {meta['model_name']}  |  predicts failure within {meta['horizon_h']} h  |  data as of {meta['as_of']}")

tab1, tab2, tab3, tab4 = st.tabs(["Fleet overview", "Machine detail", "Model performance", "Alerts"])

with tab1:
    c = st.columns(4)
    c[0].metric("Machines", len(scores))
    c[1].metric("High risk", int((scores.risk_level == "High").sum()))
    c[2].metric("Medium risk", int((scores.risk_level == "Medium").sum()))
    c[3].metric("Low risk", int((scores.risk_level == "Low").sum()))
    show = scores.copy()
    show["risk_level"] = show["risk_level"].map(ICON)
    show = show.drop(columns=["timestamp"])
    st.dataframe(show, hide_index=True, column_config={
        "risk_score": st.column_config.ProgressColumn("risk score", min_value=0.0, max_value=1.0, format="%.2f"),
        "main_driver": "main driver",
        "hours_since_maintenance": "h since maintenance"})
    st.caption("Machines marked 'Stopped' have had no reading for 3+ hours (repair or maintenance).")

with tab2:
    machine = st.selectbox("Machine", scores["machine_id"].tolist())
    h = hist[hist["machine_id"] == machine].set_index("timestamp").copy()
    h["alert threshold"] = meta["threshold"]
    st.subheader("Risk score (last 14 days)")
    st.line_chart(h[["risk_score", "alert threshold"]])
    a, b, d = st.columns(3)
    a.caption("Temperature")
    a.line_chart(h["temperature"])
    b.caption("Vibration")
    b.line_chart(h["vibration"])
    d.caption("Pressure")
    d.line_chart(h["pressure"])

with tab3:
    m = metrics["selected_test"]
    st.write(f"Selected model: **{metrics['model_name']}** (chosen on validation PR-AUC; "
             f"alert threshold tuned for ≥{int(metrics['recall_target']*100)}% recall on validation data). "
             "All numbers below are on the untouched, most recent test period.")
    c = st.columns(5)
    c[0].metric("Precision", f"{m['precision']:.2f}")
    c[1].metric("Recall", f"{m['recall']:.2f}")
    c[2].metric("PR-AUC", f"{m['pr_auc']:.2f}")
    c[3].metric("Failures caught", f"{m['failures_caught']}/{m['failures_in_test']}")
    c[4].metric("False alarms / machine-week", f"{m['false_alarms_per_machine_week']:.2f}")
    st.dataframe(pd.DataFrame(metrics["comparison"]).round(3), hide_index=True)
    for name in ["pr_curves.png", "feature_importance.png", "risk_timeline.png", "confusion_matrix.png"]:
        if (FIG_DIR / name).exists():
            st.image(str(FIG_DIR / name))

with tab4:
    st.write("High-risk machines trigger Slack and/or e-mail alerts when credentials are configured "
             "(see README). Without credentials the app runs in safe **dry-run** mode.")
    st.code(alerts.build_message(scores, meta["as_of"]))
    if st.button("Send alerts now"):
        res = alerts.dispatch(scores, meta["as_of"])
        st.success(f"Mode: {'dry-run' if res['dry_run'] else 'live'} | Slack: {res['slack']} | E-mail: {res['email']}")
    log = DATA_DIR / "alerts_log.csv"
    if log.exists():
        st.subheader("Alert log")
        st.dataframe(pd.read_csv(log).tail(20), hide_index=True)
