from typing import Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Incoming payload with model features."""

    features: Dict[str, float] = Field(
        ...,
        description="Feature name->value",
    )


class PredictResponse(BaseModel):
    """Successful prediction response."""

    risk_score: float
    model_version: str


class ErrorResponse(BaseModel):
    """Standard error response shape."""

    detail: str
    errors: List[str] = Field(default_factory=list)
