"""Request/response shapes. Pydantic validates input for us, so bad requests
(empty text, huge batches) are rejected with a clear 422 error before they
ever reach the model."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000,
                      examples=["yar service bohat slow hai"])


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)


class Prediction(BaseModel):
    text: str
    label: str                      # "negative" | "neutral" | "positive"
    confidence: float               # probability of the chosen label
    scores: dict[str, float]        # probability of every label


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
