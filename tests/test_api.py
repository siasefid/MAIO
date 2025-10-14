from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_predict_ok():
    payload = {"features": {"age":0.03,"sex":-0.04,"bmi":0.06,"bp":-0.03,"s1":-0.001,"s2":0.002,"s3":-0.004,"s4":0.01,"s5":0.02,"s6":-0.015}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "risk_score" in body and "model_version" in body and "rmse" in body

def test_predict_missing_field():
    bad = {"features": {"age":0.03}}  # missing others
    r = client.post("/predict", json=bad)
    assert r.status_code in (400, 422)
