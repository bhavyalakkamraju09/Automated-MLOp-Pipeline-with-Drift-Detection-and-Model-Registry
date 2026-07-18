"""API smoke tests for the FastAPI churn serving app.

Run just the smoke-marked test from the Airflow training DAG:
  pytest tests/test_api.py -k smoke --timeout=30
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.serving.app import app
    return TestClient(app)


def test_health_endpoint_smoke(client):
    """smoke: service boots and reports a served model version."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_predict_endpoint_shape(client):
    payload = {
        "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
        "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "TechSupport": "No", "MultipleLines": "No", "SeniorCitizen": 0,
        "Partner": 1, "Dependents": 0, "PhoneService": 1, "PaperlessBilling": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_predicted"], bool)
