"""Evidently drift detection: PSI-based feature drift + prediction drift.

Compares a reference dataset (frozen at training time) against current
production traffic, logs a drift report as an MLflow artifact, and
returns a flat metrics dict the Airflow drift_monitor DAG can threshold.
"""
import os

import mlflow
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
from evidently.report import Report

with open("params.yaml") as f:
    monitor_params = yaml.safe_load(f)["monitoring"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "PaymentMethod", "InternetService"]


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                      run_id: str = None) -> dict:
    column_mapping = ColumnMapping(
        target="Churn",
        prediction="prediction" if "prediction" in current.columns else None,
        numerical_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    metrics_list = [DataDriftPreset()]
    if "prediction" in current.columns and "Churn" in current.columns:
        metrics_list.append(ClassificationPreset())

    report = Report(metrics=metrics_list)
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    result = report.as_dict()

    drift_result = result["metrics"][0]["result"]
    drift_metrics = {
        "dataset_drift": drift_result.get("dataset_drift", False),
        "n_drifted_features": drift_result.get("number_of_drifted_columns", 0),
        "share_drifted": drift_result.get("share_of_drifted_columns", 0.0),
    }

    for col, v in drift_result.get("drift_by_columns", {}).items():
        drift_metrics[f"psi_{col}"] = v.get("stattest_threshold", v.get("drift_score", 0))

    os.makedirs("monitoring/reports", exist_ok=True)
    report_path = f"monitoring/reports/drift_{run_id or 'latest'}.html"
    report.save_html(report_path)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k: v for k, v in drift_metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(report_path, "drift_reports")

    return drift_metrics


def check_alert_threshold(drift_metrics: dict) -> tuple:
    psi_values = [v for k, v in drift_metrics.items()
                  if k.startswith("psi_") and isinstance(v, (int, float))]
    psi_max = max(psi_values, default=0)
    alert = psi_max > monitor_params["psi_threshold"]     # > 0.20
    critical = psi_max > monitor_params["psi_critical"]   # > 0.25
    return alert, critical


if __name__ == "__main__":
    reference = pd.read_parquet("data/reference/reference.parquet")
    current = pd.read_parquet("data/processed/test.parquet").sample(
        min(200, len(reference)), random_state=1
    )
    metrics = run_drift_report(reference, current)
    alert, critical = check_alert_threshold(metrics)
    print(f"Drift alert={alert}, critical={critical}")
    print(metrics)
