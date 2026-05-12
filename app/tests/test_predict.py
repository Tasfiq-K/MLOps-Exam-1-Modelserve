# ============================================================================
# ModelServe — Tests
# ============================================================================
# TODO: Write tests for the inference service.
#
# Required tests:
#   - /health returns 200 with status and model_version fields
#   - /predict returns a valid prediction response for a known entity
#   - /predict returns 400 or 422 for invalid input
#   - /metrics returns 200 with Prometheus-format text
#   - At least one test that mocks the MLflow model and verifies prediction logic
#
# Testing tools:
#   - Use FastAPI's TestClient (from fastapi.testclient import TestClient)
#   - Use unittest.mock to mock MLflow and Feast dependencies
#   - Use pytest as the test runner
#
# These tests must pass in GitHub Actions CI.
# The TA will also run them during the demo.
# ============================================================================


# ============================================================================
# ModelServe — Tests
# ============================================================================

from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient

from app.main import app


# ----------------------------------------------------------------------------
# TEST CLIENT
# ----------------------------------------------------------------------------

client = TestClient(app)


# ----------------------------------------------------------------------------
# MOCK DATA
# ----------------------------------------------------------------------------

MOCK_FEATURES = {
    "amt": 120.5,
    "amt_log": 4.79,
    "lat": 40.12,
    "long": -73.44,
    "city_pop": 500000,
    "hour": 14,
    "day": 22,
    "month": 7,
    "merch_lat": 40.20,
    "merch_long": -73.50,
}


# ----------------------------------------------------------------------------
# HEALTH ENDPOINT
# ----------------------------------------------------------------------------

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_version" in data

    assert data["status"] in [
        "healthy",
        "unhealthy",
    ]


# ----------------------------------------------------------------------------
# METRICS ENDPOINT
# ----------------------------------------------------------------------------

def test_metrics_endpoint():

    response = client.get("/metrics")

    assert response.status_code == 200

    text = response.text

    assert (
        "prediction_requests_total"
        in text
    )

    assert (
        "prediction_duration_seconds"
        in text
    )

    assert (
        "prediction_errors_total"
        in text
    )


# ----------------------------------------------------------------------------
# INVALID REQUEST
# ----------------------------------------------------------------------------

def test_predict_invalid_input():

    response = client.post(
        "/predict",
        json={"entity_id": "invalid"},
    )

    # Pydantic validation error
    assert response.status_code == 422


# ----------------------------------------------------------------------------
# MOCKED PREDICTION
# ----------------------------------------------------------------------------

@patch("app.main.feature_client")
@patch("app.main.model_service")
def test_predict_success(
    mock_model_service,
    mock_feature_client,
):

    # ------------------------------------------------------------------------
    # Mock Feast response
    # ------------------------------------------------------------------------

    mock_feature_client.get_features.return_value = (
        MOCK_FEATURES
    )

    mock_feature_client.get_features_dataframe.return_value = (
        MagicMock()
    )

    # ------------------------------------------------------------------------
    # Mock model predictions
    # ------------------------------------------------------------------------

    mock_model_service.predict.return_value = [1]

    mock_model_service.predict_proba.return_value = [
        [0.02, 0.98]
    ]

    mock_model_service.model_version = "1"

    # ------------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------------

    response = client.post(
        "/predict",
        json={
            "entity_id": 123456789
        },
    )

    assert response.status_code == 200

    data = response.json()

    # ------------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------------

    assert data["prediction"] == 1

    assert (
        abs(data["probability"] - 0.98)
        < 1e-6
    )

    assert (
        data["model_version"] == "1"
    )

    assert "timestamp" in data


# ----------------------------------------------------------------------------
# EXPLAIN ENDPOINT
# ----------------------------------------------------------------------------

@patch("app.main.feature_client")
@patch("app.main.model_service")
def test_predict_explain(
    mock_model_service,
    mock_feature_client,
):

    mock_feature_client.get_features.return_value = (
        MOCK_FEATURES
    )

    mock_feature_client.get_features_dataframe.return_value = (
        MagicMock()
    )

    mock_model_service.predict.return_value = [0]

    mock_model_service.predict_proba.return_value = [
        [0.91, 0.09]
    ]

    mock_model_service.model_version = "2"

    response = client.get(
        "/predict/123456789?explain=true"
    )

    assert response.status_code == 200

    data = response.json()

    assert "features" in data

    assert (
        data["features"]["amt"]
        == MOCK_FEATURES["amt"]
    )

    assert data["prediction"] == 0


# ----------------------------------------------------------------------------
# FEATURE STORE FAILURE
# ----------------------------------------------------------------------------

@patch("app.main.feature_client")
def test_predict_feature_failure(
    mock_feature_client,
):

    mock_feature_client.get_features.side_effect = (
        Exception("Feast unavailable")
    )

    response = client.post(
        "/predict",
        json={
            "entity_id": 999999
        },
    )

    assert response.status_code == 500


# ----------------------------------------------------------------------------
# MODEL FAILURE
# ----------------------------------------------------------------------------

@patch("app.main.feature_client")
@patch("app.main.model_service")
def test_predict_model_failure(
    mock_model_service,
    mock_feature_client,
):

    mock_feature_client.get_features.return_value = (
        MOCK_FEATURES
    )

    mock_feature_client.get_features_dataframe.return_value = (
        MagicMock()
    )

    mock_model_service.predict.side_effect = (
        Exception("Model inference failed")
    )

    response = client.post(
        "/predict",
        json={
            "entity_id": 123456789
        },
    )

    assert response.status_code == 500