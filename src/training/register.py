"""MLflow model registry lifecycle management.

None -> Staging (automatic after training + smoke test)
Staging -> Production (automatic if AUC-ROC >= min_auc, else manual gate)
Production -> Archived (when a new model is promoted)
"""
import os

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "churn_xgboost")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def promote_to_staging():
    """Find latest registered model version and transition to Staging."""
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if not versions:
        raise ValueError("No new model version found in None stage")
    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest.version, stage="Staging"
    )
    print(f"Model v{latest.version} promoted to Staging")
    return latest.version


def promote_to_production(min_auc: float = 0.85):
    """Promote Staging model to Production if it meets the quality gate."""
    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        raise ValueError("No Staging model found")
    staging = staging_versions[0]

    run = mlflow.get_run(staging.run_id)
    auc_roc = run.data.metrics.get("auc_roc", 0)
    if auc_roc < min_auc:
        raise ValueError(f"Staging model AUC-ROC {auc_roc:.4f} < threshold {min_auc}")

    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Archived"
        )

    client.transition_model_version_stage(
        name=MODEL_NAME, version=staging.version, stage="Production"
    )
    print(f"Model v{staging.version} (AUC-ROC={auc_roc:.4f}) promoted to Production")
    return staging.version


def rollback_to_staging():
    """Fallback path: if the Production model fails a health check, demote
    it back to Staging so serving can fall back to the last-known-good
    Staging version (per model_loader's fallback logic)."""
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Archived"
        )
        print(f"Rolled back Production v{v.version} to Archived")


if __name__ == "__main__":
    promote_to_staging()
