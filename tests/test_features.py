"""Unit tests for the feature engineering ColumnTransformer."""
import pandas as pd
import pytest

from src.data.features import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "tenure": [1, 24, 60],
        "MonthlyCharges": [29.85, 56.95, 99.9],
        "TotalCharges": [29.85, 1366.8, 5999.0],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": [1, 0, 1],
        "Dependents": [0, 0, 1],
        "PhoneService": [1, 1, 0],
        "PaperlessBilling": [1, 0, 1],
    })


def test_preprocessor_output_shape(sample_df):
    preprocessor = build_preprocessor()
    X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    transformed = preprocessor.fit_transform(X)
    expected_cols = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(BINARY_FEATURES)
    assert transformed.shape == (3, expected_cols)


def test_preprocessor_handles_unknown_category(sample_df):
    preprocessor = build_preprocessor()
    X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES]
    preprocessor.fit(X)
    unseen = X.copy()
    unseen.loc[0, "Contract"] = "Never-before-seen-plan"
    # Should not raise — unknown categories map to -1
    preprocessor.transform(unseen)
