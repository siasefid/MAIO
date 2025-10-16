from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from .model import predict_single

MODEL_PATH = Path("artifacts/model_baseline.pkl")

app = FastAPI(
    title="Diabetes Progression Risk Service",
    version="v0.2",
    description="Predict short-term diabetes progression risk score."
)

class PredictRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v0.2"}

@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail="Model not found.")
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([req.dict()])
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}
