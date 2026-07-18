"""Optuna Bayesian hyperparameter search (50 trials) with MLflow logging.

Writes the winning params back into params.yaml under `train:` so DVC
picks them up as the pipeline's tracked hyperparameters.
"""
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }
    model = xgb.XGBClassifier(**params, tree_method="hist", eval_metric="auc")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def run_hpo(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    mlflow.set_experiment("churn_hpo")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val), n_trials=n_trials)

    with mlflow.start_run(run_name="optuna_hpo_summary"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metric("best_auc_roc", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    print(f"Best AUC-ROC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def run_hpo_from_parquet(n_trials: int = 50) -> dict:
    """Convenience wrapper used by the Airflow training DAG."""
    train_df = pd.read_parquet("data/processed/train.parquet")
    X, y = train_df.drop("Churn", axis=1), train_df["Churn"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best_params = run_hpo(X_train, y_train, X_val, y_val, n_trials=n_trials)

    with open("params.yaml") as f:
        all_params = yaml.safe_load(f)
    all_params["train"].update(best_params)
    with open("params.yaml", "w") as f:
        yaml.safe_dump(all_params, f, sort_keys=False)

    return best_params


if __name__ == "__main__":
    run_hpo_from_parquet()
