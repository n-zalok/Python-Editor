from pydantic import BaseModel
from typing import Any, Dict


class SubmissionCreate(BaseModel):
    raw_code: str


class SubmissionCreateResponse(BaseModel):
    id: int
    raw_code: str
    recommendation_text: str


class SubmissionResult(BaseModel):
    raw_code: str
    recommendation_text: str


class SubmissionRead(BaseModel):
    id: int
    raw_code: str
    extracted_features: Dict[str, Any]
    prediction_output: float
    model_version: str
    created_at: str

    class Config:
        orm_mode = True
