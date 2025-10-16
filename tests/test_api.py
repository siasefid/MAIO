from risk_service.data import load_dataset
from risk_service.model import train_and_eval_models
from fastapi.testclient import TestClient
from risk_service.api import app
import joblib
import pandas as pd

def test_pipeline_predict(tmp_path):
    X, y = load_dataset(as_frame=True)
    rmse, model_path = train_and_eval_models(X, y, out_dir=tmp_path)
    model = joblib.load(model_path)
    feats = X.columns

    sample = X.iloc[[0]][feats].to_dict(orient="records")[0]
    client = TestClient(app)
    response = client.post("/predict", json=sample)

    assert response.status_code == 200
    assert "prediction" in response.json()

