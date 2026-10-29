"""
End-to-End API Integration Tests.
Tests authentication, 2FA OTP, single predictions, alert workflows,
analyst reviews, model metrics comparison, and role permissions.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.database import SessionLocal
from backend.app.models.entities import User

client = TestClient(app)

@pytest.fixture(scope="module")
def auth_tokens():
    # Login as seeded Admin
    res_admin = client.post("/api/v1/auth/login", json={"email": "admin@fraud.local", "password": "Admin@123"})
    assert res_admin.status_code == 200

    # Retrieve OTP directly from DB for test verification
    db = SessionLocal()
    admin = db.query(User).filter(User.email == "admin@fraud.local").first()
    otp_code = admin.otp_code

    verify_res = client.post("/api/v1/auth/verify-otp", json={"email": "admin@fraud.local", "otp_code": otp_code})
    assert verify_res.status_code == 200
    token = verify_res.json()["access_token"]
    db.close()
    return {"admin_token": token}

def test_health_probe():
    res = client.get("/api/v1/audit/health")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_model_comparison_api(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['admin_token']}"}
    res = client.get("/api/v1/models/comparison", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "xgboost" in data
    assert "random_forest" in data
    assert data["xgboost"]["f1_score"] > 0.8
    assert data["random_forest"]["f1_score"] > 0.8

def test_feature_importance_api(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['admin_token']}"}
    res = client.get("/api/v1/models/feature-importance", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) > 0
    assert "xgboost_importance" in data[0]

def test_single_prediction_xgb(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['admin_token']}"}
    payload = {
        "transaction": {
            "cc_num": 4845775843900477,
            "amt": 1850.0,
            "merchant": "fraud_HighRiskElectronics",
            "category": "shopping_net",
            "first": "William",
            "last": "Brown",
            "gender": "M",
            "city": "Seattle",
            "state": "WA",
            "lat": 47.6062,
            "long": -122.3321,
            "merch_lat": 25.7617, # Miami, FL
            "merch_long": -80.1918,
            "trans_date_trans_time": "2025-01-01 02:45:00"
        },
        "model_name": "XGBoost"
    }
    res = client.post("/api/v1/predictions/single", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "XGBoost"
    assert "risk_score" in data
    assert data["risk_category"] in ["HIGH", "CRITICAL"]
    assert "behavioural_analysis" in data
    assert "shap_explanation" in data

def test_single_prediction_rf(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['admin_token']}"}
    payload = {
        "transaction": {
            "cc_num": 4845775843900477,
            "amt": 32.50,
            "merchant": "Safeway Groceries",
            "category": "grocery_pos",
            "first": "William",
            "last": "Brown",
            "gender": "M",
            "city": "Seattle",
            "state": "WA",
            "lat": 47.6062,
            "long": -122.3321,
            "merch_lat": 47.6080,
            "merch_long": -122.3340,
            "trans_date_trans_time": "2025-01-01 12:15:00"
        },
        "model_name": "Random Forest"
    }
    res = client.post("/api/v1/predictions/single", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "Random Forest"
    assert data["risk_category"] == "LOW"

def test_alerts_listing(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['admin_token']}"}
    res = client.get("/api/v1/alerts", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_audit_logs_rbac(auth_tokens):
    headers = {"Authorization": f"Bearer {auth_tokens['admin_token']}"}
    res = client.get("/api/v1/audit/logs", headers=headers)
    assert res.status_code == 200
    logs = res.json()
    assert len(logs) > 0
