from pydantic import BaseModel
from typing import Optional, Dict, Any


class PredictionRequest(BaseModel):
    entity_id: int


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    timestamp: str


class ExplainPredictionResponse(PredictionResponse):
    features: Dict[str, Any]