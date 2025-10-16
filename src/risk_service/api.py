from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from risk_service.model import load_model
from risk_service.schemas import PredictRequest, PredictResponse, ErrorResponse
import pandas as pd
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model_baseline.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v0.1")

app = FastAPI(title="Diabetes Progression Risk Service", version=MODEL_VERSION)

# Load on startup
pipeline, feature_order = load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.features])
        # Validate feature names
        missing = set(feature_order) - set(df.columns)
        extra = set(df.columns) - set(feature_order)
        if missing:
            return JSONResponse(status_code=400, content={"detail": "Bad input", "errors": [f"Missing: {sorted(missing)}"]})
        if extra:
            return JSONResponse(status_code=400, content={"detail": "Bad input", "errors": [f"Unknown: {sorted(extra)}"]})
        df = df[feature_order]
        score = float(pipeline.predict(df)[0])
        return PredictResponse(risk_score=score, model_version=MODEL_VERSION)
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": "Prediction failed", "errors": [str(e)]})
