"""FastAPI service for Roman Urdu sentiment.

Run locally:  uvicorn app.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.model import get_model
from app.schemas import (BatchPredictRequest, BatchPredictResponse,
                         Prediction, PredictRequest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup so the first user does not wait.
    # Tests set PRELOAD_MODEL=0 so they can run without the real model.
    if os.getenv("PRELOAD_MODEL", "1") == "1":
        get_model()
    yield


app = FastAPI(
    title="Roman Urdu Sentiment API",
    description="Classifies Roman Urdu customer messages as negative, neutral or positive.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest, model=Depends(get_model)):
    return model.predict([req.text])[0]


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest, model=Depends(get_model)):
    return {"predictions": model.predict(req.texts)}
