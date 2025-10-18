from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import predict_single

app = FastAPI(
    title="Diabetes Progression Risk Service",
    version="v0.1",
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
    return {"status": "ok", "model_version": "v0.1"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        pred = predict_single(req.model_dump())
        return {"prediction": pred}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
