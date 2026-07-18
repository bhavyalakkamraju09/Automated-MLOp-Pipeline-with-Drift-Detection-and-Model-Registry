"""DAG 2: training_pipeline (triggered by data_pipeline, or manual/weekly)

load_features -> hyperparameter_tuning -> train_model -> register_and_stage
  -> smoke_test -> [branch: AUC gate] -> promote_to_production
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    schedule=None,  # triggered by data_pipeline DAG or manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    dvc_pull = BashOperator(
        task_id="dvc_pull_features",
        bash_command="cd /opt/airflow && dvc pull data/processed/ || true",
    )

    def _run_hpo():
        from src.training.tune import run_hpo_from_parquet
        run_hpo_from_parquet(n_trials=50)

    run_hpo = PythonOperator(task_id="hyperparameter_tuning", python_callable=_run_hpo)

    def _train():
        from src.training.train import train_model
        model, metrics = train_model()
        return metrics

    train = PythonOperator(task_id="train_model", python_callable=_train)

    def _register():
        from src.training.register import promote_to_staging
        return promote_to_staging()

    register = PythonOperator(task_id="register_and_stage", python_callable=_register)

    smoke_test = BashOperator(
        task_id="smoke_test",
        bash_command="cd /opt/airflow && python -m pytest tests/test_api.py -k smoke --timeout=30",
    )

    def check_promote(**context):
        metrics = context["task_instance"].xcom_pull(task_ids="train_model")
        if metrics and metrics.get("auc_roc", 0) > 0.85:
            return "promote_to_production"
        return "skip_promotion"

    branch = BranchPythonOperator(task_id="check_auto_promote", python_callable=check_promote)

    def _promote():
        from src.training.register import promote_to_production
        return promote_to_production()

    promote = PythonOperator(task_id="promote_to_production", python_callable=_promote)

    skip = BashOperator(
        task_id="skip_promotion",
        bash_command="echo 'AUC-ROC below threshold — held in Staging for manual review'",
    )

    dvc_pull >> run_hpo >> train >> register >> smoke_test >> branch >> [promote, skip]
