from fastapi.testclient import TestClient
from risk_service.api import app
from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models
import joblib
import pandas as pd


client = TestClient(app)


def test_pipeline_predict(tmp_path):
    X, y = load_dataset(as_frame=True)
    rmse, model_path = train_and_eval_models(X, y, out_dir=tmp_path)
    model = joblib.load(model_path)
    sample = X.iloc[[0]].to_dict(orient="records")[0]

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
