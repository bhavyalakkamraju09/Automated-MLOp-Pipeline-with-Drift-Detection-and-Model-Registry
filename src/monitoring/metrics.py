"""Prometheus metric definitions shared between the FastAPI middleware
and any batch jobs (e.g. the drift monitor DAG) that want to push
gauges without duplicating metric names/labels.
"""
from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNTER = Counter(
    "predictions_total", "Total predictions", ["model_version", "predicted_class"]
)

LATENCY_HISTOGRAM = Histogram(
    "prediction_latency_seconds", "Prediction latency",
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1],
)

CHURN_RATE_GAUGE = Gauge(
    "churn_rate_rolling", "Rolling churn rate (last 1000 predictions)"
)

MODEL_VERSION_GAUGE = Gauge(
    "model_version_info", "Current production model version", ["version", "run_id"]
)

DRIFT_PSI_GAUGE = Gauge(
    "feature_drift_psi", "Latest PSI value per feature", ["feature"]
)

DRIFT_CRITICAL_GAUGE = Gauge(
    "drift_critical", "1 if the last drift check crossed the critical threshold"
)
