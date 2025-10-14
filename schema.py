from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Union

class Features(BaseModel):
    age: float = Field(...)
    sex: float = Field(...)
    bmi: float = Field(...)
    bp: float = Field(...)
    s1: float = Field(...)
    s2: float = Field(...)
    s3: float = Field(...)
    s4: float = Field(...)
    s5: float = Field(...)
    s6: float = Field(...)

class PredictRequest(BaseModel):
    features: Union[Features, List[Features]]
