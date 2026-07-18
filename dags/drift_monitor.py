"""DAG 3: drift_monitor (hourly)

fetch_recent_predictions -> fetch_reference_data -> run_evidently_report
  -> check_psi_threshold -> alert_slack_if_drift -> trigger_retrain_if_critical
"""
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="drift_monitor",
    default_args=default_args,
    schedule="0 * * * *",  # hourly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "monitoring"],
) as dag:

    def fetch_predictions(**context):
        # In production: query the prediction log DB / a Prometheus range
        # query for recent /predict payloads. For the demo: simulate by
        # sampling from the test set with an injected shift so the alert
        # path is exercisable end-to-end.
        import numpy as np

        current = pd.read_parquet("data/processed/test.parquet").sample(200)
        current["tenure"] = current["tenure"] * np.random.normal(0.85, 0.05, len(current))
        current["MonthlyCharges"] = current["MonthlyCharges"] * np.random.normal(1.1, 0.05, len(current))
        current["prediction"] = (
            current["Churn"].values + np.random.normal(0, 0.1, len(current))
        ).clip(0, 1)
        context["ti"].xcom_push(key="current_data", value=current.to_json())

    fetch = PythonOperator(task_id="fetch_predictions", python_callable=fetch_predictions)

    def run_drift(**context):
        from src.monitoring.alerting import format_drift_alert, send_slack_alert
        from src.monitoring.drift_detector import check_alert_threshold, run_drift_report

        reference = pd.read_parquet("data/reference/reference.parquet")
        current_json = context["ti"].xcom_pull(key="current_data")
        current = pd.read_json(current_json)

        drift_metrics = run_drift_report(reference, current)
        alert, critical = check_alert_threshold(drift_metrics)

        context["ti"].xcom_push(key="drift_critical", value=critical)
        context["ti"].xcom_push(key="drift_metrics", value=drift_metrics)

        if alert:
            send_slack_alert(format_drift_alert(drift_metrics, critical))

        print(f"Drift alert={alert}, critical={critical}: {drift_metrics}")
        return critical  # ShortCircuit: only continue to retrain if critical

    drift = ShortCircuitOperator(task_id="run_drift_check", python_callable=run_drift)

    retrain = TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="training_pipeline")

    fetch >> drift >> retrain
