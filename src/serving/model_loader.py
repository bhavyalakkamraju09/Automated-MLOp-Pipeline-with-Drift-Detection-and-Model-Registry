"""Load the current Production-stage model from the MLflow registry.

Falls back to the Staging version if Production is unavailable or fails
a basic sanity check, and falls back to a local pickle (models/churn_model.pkl)
if MLflow itself is unreachable — so `uvicorn src.serving.app:app` still
boots for local dev without the full Docker Compose stack running.
"""
import os
import pickle

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "churn_xgboost")
LOCAL_FALLBACK_PATH = "models/churn_model.pkl"


def _load_stage(client, stage: str):
    import mlflow

    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if not versions:
        return None, None
    version = versions[0]
    model_uri = f"models:/{MODEL_NAME}/{stage}"
    model = mlflow.xgboost.load_model(model_uri)
    info = {"version": version.version, "run_id": version.run_id, "stage": stage}
    return model, info


def load_production_model():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        model, info = _load_stage(client, "Production")
        if model is not None:
            return model, info

        print("No Production model found — falling back to Staging")
        model, info = _load_stage(client, "Staging")
        if model is not None:
            return model, info

        raise RuntimeError("No Production or Staging model registered")

    except Exception as e:
        print(f"MLflow registry unavailable ({e}) — loading local fallback pickle")
        with open(LOCAL_FALLBACK_PATH, "rb") as f:
            model = pickle.load(f)
        info = {"version": "local-fallback", "run_id": "n/a", "stage": "local"}
        return model, info
