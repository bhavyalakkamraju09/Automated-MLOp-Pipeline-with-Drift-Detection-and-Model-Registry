"""Train the churn XGBoost model with full MLflow experiment tracking.

DVC stage: reads train/test parquet, trains, logs params/metrics/SHAP
artifact/model to MLflow, and writes metrics/eval.json for DVC's own
metrics tracking.
"""
import json
import os

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def train_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    X_train, y_train = train_df.drop("Churn", axis=1), train_df["Churn"]
    X_test, y_test = test_df.drop("Churn", axis=1), test_df["Churn"]

    mlflow.set_experiment("churn_prediction")

    model_params = {k: v for k, v in params.items() if k != "eval_metric"}

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        model = xgb.XGBClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=params["scale_pos_weight"],
            random_state=params["random_state"],
            eval_metric=params.get("eval_metric", "auc"),
            tree_method="hist",
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        metrics = {
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "auc_pr": float(average_precision_score(y_test, y_prob)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        print(f"AUC-ROC: {metrics['auc_roc']:.4f} | AUC-PR: {metrics['auc_pr']:.4f} | F1: {metrics['f1']:.4f}")

        # SHAP summary plot as artifact (optional dependency — skip gracefully)
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            os.makedirs("metrics", exist_ok=True)
            plt.savefig("metrics/shap_summary.png", bbox_inches="tight")
            plt.close()
            mlflow.log_artifact("metrics/shap_summary.png")
        except Exception as e:
            print(f"SHAP artifact skipped: {e}")

        mlflow.xgboost.log_model(model, "model", registered_model_name="churn_xgboost")

        os.makedirs("models", exist_ok=True)
        import pickle
        with open("models/churn_model.pkl", "wb") as f:
            pickle.dump(model, f)

        os.makedirs("metrics", exist_ok=True)
        with open("metrics/eval.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"MLflow run_id: {run.info.run_id}")
        return model, metrics


if __name__ == "__main__":
    train_model()
