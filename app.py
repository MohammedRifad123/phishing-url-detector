"""FastAPI backend for the Phishing URL Detector.

Loads the trained model from ``model.pkl`` and exposes:
    GET  /           - service info
    GET  /health     - liveness probe
    POST /predict    - classify a URL

Run with:
    uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from extractor import FEATURE_NAMES, extract_features

MODEL_PATH = Path(__file__).with_name("model.pkl")

# Decision thresholds on P(PHISHING).
PHISHING_THRESHOLD = 0.70
SAFE_THRESHOLD = 0.35

# Loaded in the lifespan hook below.
_state: dict[str, Any] = {"model": None, "feature_names": FEATURE_NAMES}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH.name} not found. Run `python train.py` first."
        )
    bundle = joblib.load(MODEL_PATH)
    # Accept both the dict bundle written by train.py and a bare estimator.
    if isinstance(bundle, dict) and "model" in bundle:
        _state["model"] = bundle["model"]
        _state["feature_names"] = bundle.get("feature_names", FEATURE_NAMES)
        _state["estimator_name"] = bundle.get("estimator_name", "unknown")
    else:
        _state["model"] = bundle
        _state["feature_names"] = FEATURE_NAMES
        _state["estimator_name"] = "unknown"
    print(f"[app] Loaded model ({_state['estimator_name']}) from {MODEL_PATH.name}")
    yield
    _state["model"] = None


app = FastAPI(
    title="Phishing URL Detector",
    description="Classifies URLs as SAFE, SUSPICIOUS or PHISHING.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048,
                     example="http://paypal.com@secure-login.evil.tk/verify")


class PredictResponse(BaseModel):
    url: str
    classification: str
    confidence: float
    phishing_probability: float
    features: dict[str, float]


def classify(model: Any, feature_names: list[str], url: str) -> dict[str, Any]:
    """Run the model on ``url`` and map probabilities to a 3-way label."""
    feats = extract_features(url)
    vector = np.array([[feats[name] for name in feature_names]], dtype=float)

    proba = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    if "PHISHING" in classes:
        p_phishing = float(proba[classes.index("PHISHING")])
    else:  # numeric labels, 1 = phishing
        p_phishing = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])

    if p_phishing >= PHISHING_THRESHOLD:
        classification, confidence = "PHISHING", p_phishing
    elif p_phishing <= SAFE_THRESHOLD:
        classification, confidence = "SAFE", 1.0 - p_phishing
    else:
        classification = "SUSPICIOUS"
        confidence = max(p_phishing, 1.0 - p_phishing)

    return {
        "url": url,
        "classification": classification,
        "confidence": round(confidence, 4),
        "phishing_probability": round(p_phishing, 4),
        "features": feats,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Phishing URL Detector", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if _state["model"] is not None else "model-not-loaded"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL must not be empty.")
    if "://" not in url:
        # Normalise scheme-less input so urllib can parse a hostname.
        url = "http://" + url
    if len(url) > 2048:
        raise HTTPException(status_code=400, detail="URL exceeds 2048 characters.")

    try:
        result = classify(_state["model"], _state["feature_names"], url)
    except Exception as exc:  # noqa: BLE001 - surface any parse/model issue cleanly
        raise HTTPException(status_code=422, detail=f"Could not analyse URL: {exc}") from exc

    return PredictResponse(**result)
