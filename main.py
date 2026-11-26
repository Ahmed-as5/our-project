from contextlib import asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel, Field


MODEL_PATH = os.getenv("MODEL_PATH", "models/pipeline.joblib")

DEPARTMENT_MAP: dict[str, str] = {
    "business": "Finance & Corporate Strategy",
    "entertainment": "Brand, Media & Communications",
    "politics": "Public Policy & Legal Affairs",
    "sport": "Events, Sponsorships & Community",
    "tech": "Engineering & Product Operations",
}

DEFAULT_DEPARTMENT = "General Operations & Triage"


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=250_000, description="Document content to classify")


class ClassifyResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggestion: str


def compute_confidence(model: Any, text: str) -> float:
    """Return a probability-like confidence score in [0.0, 1.0]."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        return float(np.max(probs))

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([text])[0], dtype=float)
        if scores.ndim == 0:
            # Binary margin -> sigmoid
            return float(1.0 / (1.0 + np.exp(-scores)))
        # Multi-class margins -> softmax max
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)
        return float(np.max(probs))

    return 1.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipeline = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {exc}") from exc
    yield
    app.state.pipeline = None


app = FastAPI(
    title="DocClassifier API",
    version="1.0.0",
    lifespan=lifespan,
)

raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = (
    ["*"]
    if raw_origins.strip() == "*"
    else [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    loaded = getattr(app.state, "pipeline", None) is not None
    return {"status": "ok" if loaded else "degraded"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_document(payload: ClassifyRequest) -> ClassifyResponse:
    cleaned_text = payload.text.strip()
    if not cleaned_text:
        raise HTTPException(status_code=422, detail="Document text cannot be empty or whitespace only.")

    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Classification model is not loaded.")

    try:
        predicted_label = str(pipeline.predict([cleaned_text])[0])
        confidence = compute_confidence(pipeline, cleaned_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    department = DEPARTMENT_MAP.get(predicted_label.strip().lower(), DEFAULT_DEPARTMENT)

    return ClassifyResponse(
        label=predicted_label,
        confidence=round(confidence, 4),
        suggestion=department,
    )