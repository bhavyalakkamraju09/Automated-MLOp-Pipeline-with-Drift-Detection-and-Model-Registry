"""Feature engineering: ColumnTransformer pipeline, DVC stage.

Reads validated parquet, builds train/test splits, fits a
StandardScaler + OrdinalEncoder ColumnTransformer, and writes
transformed parquet files plus the fitted preprocessor artifact.
"""
import os
import pickle

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

with open("params.yaml") as f:
    params = yaml.safe_load(f)["featurize"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "Contract", "PaymentMethod", "InternetService",
    "OnlineSecurity", "TechSupport", "MultipleLines",
]
BINARY_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
    ], remainder="drop")


def run_feature_pipeline():
    df = pd.read_parquet("data/processed/validated.parquet")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Normalize binary Yes/No columns to 0/1 so downstream dtypes are numeric
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if df[col].dtype == object:
            df[col] = (df[col] == "Yes").astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], stratify=y,
        random_state=params["random_state"],
    )

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    pd.DataFrame(X_train_t, columns=feature_names).assign(
        Churn=y_train.values
    ).to_parquet("data/processed/train.parquet")

    pd.DataFrame(X_test_t, columns=feature_names).assign(
        Churn=y_test.values
    ).to_parquet("data/processed/test.parquet")

    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # Also stash a reference sample for Evidently drift baseline
    os.makedirs("data/reference", exist_ok=True)
    pd.DataFrame(X_train_t, columns=feature_names).assign(
        Churn=y_train.values
    ).sample(min(500, len(X_train)), random_state=42).to_parquet(
        "data/reference/reference.parquet"
    )

    print(f"Features: {len(feature_names)} | Train: {len(X_train)} | Test: {len(X_test)}")


if __name__ == "__main__":
    run_feature_pipeline()
