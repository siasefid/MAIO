from __future__ import annotations
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import json, joblib, pandas as pd, os
from src.schema import PredictRequest, Features
from src.data import FEATURES

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

app = FastAPI(title="Diabetes Progression Risk API", version="0.1")

def _load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)):
        raise FileNotFoundError("Model or metrics not found. Train first.")
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return model, metrics

model, metrics = _load_artifacts()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"model_version": metrics.get("model_version"), "rmse": metrics.get("rmse")}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        rows = req.features if isinstance(req.features, list) else [req.features]
        df = pd.DataFrame([r.model_dump() for r in rows], columns=FEATURES)
        preds = model.predict(df).tolist()
        out = {
            "risk_score": preds[0] if len(preds) == 1 else preds,
            "model_version": metrics["model_version"],
            "rmse": metrics["rmse"]
        }
        return out
    except KeyError as e:
        missing = str(e).strip("'")
        raise HTTPException(status_code=400, detail={
            "error": "ValidationError",
            "message": f"Missing feature: {missing}",
            "required_features": FEATURES
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "ModelError",
            "message": str(e),
            "required_features": FEATURES
        })

@app.exception_handler(Exception)
async def fallback_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "ServerError", "message": str(exc)})
