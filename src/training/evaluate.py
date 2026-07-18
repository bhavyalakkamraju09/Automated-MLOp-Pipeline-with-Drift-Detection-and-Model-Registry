"""Standalone evaluation utility: AUC-ROC, AUC-PR, F1, confusion matrix.

Useful for re-scoring a saved model against a fresh test slice without
retraining (e.g. from the Airflow smoke test task).
"""
import json
import pickle

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate(model_path: str = "models/churn_model.pkl",
             test_path: str = "data/processed/test.parquet") -> dict:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_parquet(test_path)
    X_test, y_test = test_df.drop("Churn", axis=1), test_df["Churn"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "auc_pr": float(average_precision_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }
    return metrics


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2))
