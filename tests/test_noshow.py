import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from noshow_iq.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200():
    payload = {
        "age": 30,
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 1,
        "days_in_advance": 5,
        "appt_day_of_week": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_has_risk_level():
    payload = {
        "age": 45,
        "scholarship": 1,
        "hypertension": 1,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "days_in_advance": 10,
        "appt_day_of_week": 3,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert "risk_level" in data
    assert data["risk_level"] in ["high", "low"]


def test_predict_has_probability():
    payload = {
        "age": 25,
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 1,
        "days_in_advance": 2,
        "appt_day_of_week": 1,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_has_recommendation():
    payload = {
        "age": 60,
        "scholarship": 0,
        "hypertension": 1,
        "diabetes": 1,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "days_in_advance": 7,
        "appt_day_of_week": 4,
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert "recommendation" in data


def test_history_returns_list():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_stats_returns_correct_keys():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "high_risk_count" in data
    assert "low_risk_count" in data
    assert "average_probability" in data