"""Stage 3a: Model API (FastAPI).

Serves the model packaged in stage 2. Runs in its own Docker container.

Endpoints
    GET  /health   liveness probe
    GET  /meta     input schema + test metrics (the web app builds its form from this)
    POST /predict  prediction for one district
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import build_features

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

app = FastAPI(
    title="California Housing Price API",
    description="Model API for PMLDL Assignment 1",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)
meta: dict[str, Any] = json.loads(META_PATH.read_text())


class HousingInput(BaseModel):
    """One California census district."""

    longitude: float = Field(..., examples=[-118.49])
    latitude: float = Field(..., examples=[34.26])
    housing_median_age: float = Field(..., ge=0, examples=[29.0])
    total_rooms: float = Field(..., gt=0, examples=[2127.0])
    total_bedrooms: float = Field(..., gt=0, examples=[435.0])
    population: float = Field(..., gt=0, examples=[1166.0])
    households: float = Field(..., gt=0, examples=[409.0])
    median_income: float = Field(..., gt=0, examples=[3.53])
    ocean_proximity: str = Field(..., examples=["<1H OCEAN"])


class Prediction(BaseModel):
    prediction: float
    target: str
    unit: str = "USD"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": meta.get("model_name", "unknown")}


@app.get("/meta")
def get_meta() -> dict[str, Any]:
    """Input schema and quality metrics of the currently deployed model."""
    return meta


@app.post("/predict", response_model=Prediction)
def predict(payload: HousingInput) -> Prediction:
    raw = pd.DataFrame([payload.model_dump()])
    try:
        features = build_features(raw)
        value = float(model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
    return Prediction(prediction=value, target=meta.get("target", "target"))
