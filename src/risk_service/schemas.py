from pydantic import BaseModel, Field
from typing import Dict, List

class PredictRequest(BaseModel):
    # features: dict where keys are feature names; values numeric
    features: Dict[str, float] = Field(..., description="Feature name->value")

class PredictResponse(BaseModel):
    risk_score: float
    model_version: str

class ErrorResponse(BaseModel):
    detail: str
    errors: List[str] = []
