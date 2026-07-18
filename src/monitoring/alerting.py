"""PSI threshold check + Slack webhook alert helper.

Called from the Airflow drift_monitor DAG after run_drift_report().
Falls back to a no-op if SLACK_WEBHOOK_URL isn't configured, so local
runs without Slack set up don't fail the DAG.
"""
import json
import os
import urllib.request

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def send_slack_alert(message: str) -> bool:
    if not SLACK_WEBHOOK_URL or "your_webhook_here" in SLACK_WEBHOOK_URL:
        print(f"[alerting] SLACK_WEBHOOK_URL not configured — would have sent: {message}")
        return False
    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"[alerting] Slack alert failed: {e}")
        return False


def format_drift_alert(drift_metrics: dict, critical: bool) -> str:
    severity = ":rotating_light: CRITICAL" if critical else ":warning: WARNING"
    psi_lines = "\n".join(
        f"  - {k}: {v:.3f}" for k, v in drift_metrics.items()
        if k.startswith("psi_") and isinstance(v, (int, float))
    )
    return (
        f"{severity} — feature drift detected in churn_xgboost\n"
        f"Dataset drift: {drift_metrics.get('dataset_drift')}\n"
        f"Drifted features: {drift_metrics.get('n_drifted_features')}\n"
        f"{psi_lines}"
    )
