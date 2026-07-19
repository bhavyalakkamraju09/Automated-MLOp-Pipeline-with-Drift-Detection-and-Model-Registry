import os
import pickle

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "churn_xgboost")
LOCAL_FALLBACK_PATH = "models/churn_model.pkl"


def _load_stage(client, stage: str):
    import mlflow

    # NOTE: get_latest_versions(stages=[...]) is deprecated and unreliable
    # against this MLflow server version — it can return empty even when
    # search_model_versions confirms a version is in that exact stage.
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = [v for v in all_versions if v.current_stage == stage]
    if not matching:
        return None, None
    version = max(matching, key=lambda v: int(v.version))
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
