"""Send alerts for high-risk machines by Slack and/or e-mail.

Configure with environment variables (or Streamlit secrets / GitHub secrets):
  SLACK_WEBHOOK_URL
  SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, ALERT_FROM, ALERT_TO
If nothing is configured the script runs in DRY-RUN mode: it prints the message
and writes it to data/alerts_log.csv, so you can always test it safely.
"""
import json
import os
import smtplib
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage

import pandas as pd

from .config import DATA_DIR


def high_risk(scores):
    return scores[scores["risk_level"] == "High"]


def build_message(scores, as_of=""):
    high = high_risk(scores)
    if high.empty:
        return "No machines are currently at high risk."
    lines = [f"Predictive maintenance alert - {len(high)} machine(s) at HIGH risk of failing within 24 h"
             + (f" (data as of {as_of})" if as_of else ""), ""]
    for _, r in high.iterrows():
        lines.append(f"- {r['machine_id']} ({r['machine_type']}): risk {r['risk_score']:.2f}, "
                     f"{r['main_driver']}, {int(r['hours_since_maintenance'])} h since last maintenance")
    lines += ["", "Recommended action: schedule an inspection for these machines."]
    return "\n".join(lines)


def send_slack(text, webhook):
    req = urllib.request.Request(webhook, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return 200 <= resp.status < 300


def send_email(subject, body):
    host, to = os.getenv("SMTP_HOST"), os.getenv("ALERT_TO")
    msg = EmailMessage()
    msg["Subject"], msg["To"] = subject, to
    msg["From"] = os.getenv("ALERT_FROM") or os.getenv("SMTP_USER") or "alerts@localhost"
    msg.set_content(body)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=15) as s:
        s.starttls(context=ssl.create_default_context())
        if os.getenv("SMTP_USER"):
            s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
        s.send_message(msg)
    return True


def dispatch(scores=None, as_of="", dry_run=None):
    """Send (or simulate) alerts. Returns a dict describing what happened."""
    if scores is None:
        scores = pd.read_csv(DATA_DIR / "risk_scores.csv")
    message = build_message(scores, as_of)
    n_high = len(high_risk(scores))
    slack, email = os.getenv("SLACK_WEBHOOK_URL"), (os.getenv("SMTP_HOST") and os.getenv("ALERT_TO"))
    if dry_run is None:
        dry_run = not (slack or email)
    result = {"high_risk_machines": n_high, "dry_run": bool(dry_run), "slack": None, "email": None, "message": message}

    if n_high and not dry_run:
        if slack:
            try:
                result["slack"] = send_slack(message, slack)
            except Exception as exc:  # noqa: BLE001
                result["slack"] = f"failed: {exc}"
        if email:
            try:
                result["email"] = send_email(f"[ALERT] {n_high} machine(s) at high risk", message)
            except Exception as exc:  # noqa: BLE001
                result["email"] = f"failed: {exc}"

    log = pd.DataFrame([{"sent_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "high_risk_machines": n_high, "mode": "dry-run" if dry_run else "live",
                         "slack": result["slack"], "email": result["email"]}])
    path = DATA_DIR / "alerts_log.csv"
    log.to_csv(path, mode="a", header=not path.exists(), index=False)
    return result


if __name__ == "__main__":
    out = dispatch(dry_run=True if "--dry-run" in sys.argv else None)
    print(out["message"])
    print(f"\n[alerts] mode={'dry-run' if out['dry_run'] else 'live'} slack={out['slack']} email={out['email']}")
