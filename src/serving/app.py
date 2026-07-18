"""FastAPI churn prediction service with Prometheus instrumentation.

Loads the Production-stage model from the MLflow registry at startup
(with Staging/local fallback — see model_loader.py) plus the fitted
sklearn preprocessor (models/preprocessor.pkl, produced by
src/data/features.py) so incoming raw request fields are transformed
the same way training data was, before scoring. Exposes:
  POST /predict   -> churn probability + prediction
  GET  /health    -> liveness + currently served model version
  GET  /metrics   -> Prometheus scrape endpoint
"""
import pickle
import time

import pandas as pd
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from src.monitoring.metrics import (
    CHURN_RATE_GAUGE,
    LATENCY_HISTOGRAM,
    MODEL_VERSION_GAUGE,
    PREDICTION_COUNTER,
)
from src.serving.middleware import PrometheusMiddleware
from src.serving.model_loader import load_production_model

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Contract", "PaymentMethod", "InternetService",
    "OnlineSecurity", "TechSupport", "MultipleLines",
]
BINARY_FEATURES = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

app = FastAPI(title="Churn Prediction Service")
app.add_middleware(PrometheusMiddleware)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

model, model_info = load_production_model()
MODEL_VERSION_GAUGE.labels(version=str(model_info["version"]), run_id=model_info["run_id"]).set(1)

try:
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    preprocessor = None
    print("WARNING: models/preprocessor.pkl not found — /predict will reject requests "
          "until `dvc repro` (or src/data/features.py) has been run at least once.")

_recent_predictions = []  # rolling window for churn rate gauge
_ROLLING_WINDOW = 1000


class ChurnRequest(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    MultipleLines: str
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int


@app.post("/predict")
async def predict(request: ChurnRequest):
    if preprocessor is None:
        return {"error": "preprocessor not loaded — run the feature pipeline first"}

    start = time.monotonic()
    raw_df = pd.DataFrame([request.model_dump()])[FEATURE_ORDER]
    transformed = preprocessor.transform(raw_df)
    df = pd.DataFrame(transformed, columns=FEATURE_ORDER)

    prob = float(model.predict_proba(df)[0, 1])
    churn = int(prob > 0.5)
    latency = time.monotonic() - start

    PREDICTION_COUNTER.labels(
        model_version=str(model_info["version"]), predicted_class=str(churn)
    ).inc()
    LATENCY_HISTOGRAM.observe(latency)

    _recent_predictions.append(churn)
    if len(_recent_predictions) > _ROLLING_WINDOW:
        _recent_predictions.pop(0)
    CHURN_RATE_GAUGE.set(sum(_recent_predictions) / len(_recent_predictions))

    return {
        "churn_probability": prob,
        "churn_predicted": bool(churn),
        "model_version": model_info["version"],
        "latency_ms": round(latency * 1000, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_info["version"], "stage": model_info.get("stage")}
