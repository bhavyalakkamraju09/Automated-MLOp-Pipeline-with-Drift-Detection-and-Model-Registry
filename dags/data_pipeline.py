"""DAG 1: data_pipeline (daily)

ingest_raw_data -> validate_schema -> run_feature_engineering
  -> dvc_add_data -> dvc_push -> trigger_training_dag (conditionally)
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="data_pipeline",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "data"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_raw_data",
        bash_command="cd /opt/airflow && python src/data/ingest.py",
    )

    validate = BashOperator(
        task_id="validate_schema",
        bash_command="cd /opt/airflow && python src/data/validate.py",
    )

    featurize = BashOperator(
        task_id="run_feature_engineering",
        bash_command="cd /opt/airflow && python src/data/features.py",
    )

    dvc_add = BashOperator(
        task_id="dvc_add_data",
        bash_command=(
            "cd /opt/airflow && "
            "dvc add data/raw/telco_churn.csv data/processed/train.parquet "
            "data/processed/test.parquet || true"
        ),
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command="cd /opt/airflow && dvc push || true",
    )

    def decide_retrain(**context):
        # In production this would check whether new data volume / drift
        # since the last training run warrants a retrain. For the demo,
        # always proceed to keep the pipeline demonstrable end-to-end.
        return "trigger_training_dag"

    branch = BranchPythonOperator(
        task_id="check_should_retrain",
        python_callable=decide_retrain,
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training_dag",
        trigger_dag_id="training_pipeline",
    )

    ingest >> validate >> featurize >> dvc_add >> dvc_push >> branch >> trigger_training
