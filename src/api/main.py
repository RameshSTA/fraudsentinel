"""
Proactive Scam Intelligence — Real-Time Inference API
======================================================

Serves calibrated fraud risk scores over a REST endpoint.

Architecture
------------
- FastAPI with Pydantic v2 input validation
- Torch MLP loaded once at startup (no per-request model load)
- Temperature scaling applied automatically if artefact is present
- Risk band assigned using the production policy (capacity-based: 1%/3%/8%)
- /health endpoint for liveness/readiness checks
- /score endpoint for real-time single-transaction scoring
- /score/batch endpoint for small offline-style batches

Usage
-----
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    # Or via Makefile:
    make serve

Interactive docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.inference.torch_infer import TorchArtifact, load_torch_artifact, predict_proba_torch

# ─── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/torch_mlp.pt")
TEMPERATURE_PATH = Path("models/torch_temperature.json")

# Capacity-based risk band thresholds (stored as top-percentile cutoffs)
# These are calibrated against the training distribution.
# In production, cutoffs should be recalibrated on each model refresh.
BAND_POLICY = {
    "critical_top_pct": 0.01,    # top 1% → Block / Immediate Review
    "high_top_pct": 0.03,        # top 1–3% → Step-Up Authentication
    "medium_top_pct": 0.08,      # top 3–8% → Monitor / Delay
}

# Score thresholds derived from test-set quantiles (from reports/risk_band_summary.json)
SCORE_CUTOFFS = {
    "critical": 0.9528,
    "high": 0.9121,
    "medium": 0.7430,
}

BAND_ACTIONS = {
    "critical": "BLOCK_AND_REVIEW — Immediate manual review required",
    "high": "STEP_UP_AUTH — Request additional authentication",
    "medium": "MONITOR — Flag for delayed review; apply additional rules",
    "low": "APPROVE — Auto-approve with standard monitoring",
}

logger = logging.getLogger("scam_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ─── App & Startup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Proactive Scam Intelligence API",
    description=(
        "Real-time fraud risk scoring with capacity-based risk banding. "
        "Returns a calibrated fraud probability and operational risk band for each transaction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global state (loaded once on startup)
_artifact: Optional[TorchArtifact] = None
_temperature: Optional[float] = None
_startup_time: float = 0.0


@app.on_event("startup")
def load_model() -> None:
    global _artifact, _temperature, _startup_time
    t0 = time.time()

    if not MODEL_PATH.exists():
        logger.error("Model artefact not found at %s. Run 'make train' first.", MODEL_PATH)
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    logger.info("Loading Torch MLP from %s", MODEL_PATH)
    _artifact = load_torch_artifact(MODEL_PATH)

    if TEMPERATURE_PATH.exists():
        with TEMPERATURE_PATH.open() as f:
            _temperature = float(json.load(f).get("temperature", 1.0))
        logger.info("Temperature scaling loaded: T=%.4f", _temperature)
    else:
        logger.warning("No temperature file found — using uncalibrated probabilities.")
        _temperature = None

    _startup_time = time.time() - t0
    logger.info("Model loaded in %.2fs. API ready.", _startup_time)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Input schema for a single transaction to be scored.

    All features must match those used during training
    (defined in models/torch_feature_schema.json).
    """

    # Identity
    TransactionID: Optional[int] = Field(None, description="Optional transaction identifier")

    # Core transaction
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount (AUD/USD)")

    # Entity keys (concatenated card/address composite — computed upstream)
    entity_key: str = Field(
        ..., description="Composite entity key: card1||card2||card3||card5||addr1"
    )
    fingerprint_key: str = Field(
        ..., description="Propagation fingerprint: DeviceInfo||P_emaildomain||ProductCD"
    )

    # Velocity features (1h window)
    cnt_1h: float = Field(..., ge=0, description="Transaction count for entity in past 1 hour")
    sum_amt_1h: float = Field(..., ge=0, description="Sum of amounts for entity in past 1 hour")

    # Velocity features (24h window)
    cnt_24h: float = Field(..., ge=0, description="Transaction count for entity in past 24 hours")
    avg_amt_24h: float = Field(..., ge=0, description="Average amount for entity in past 24 hours")

    # Behavioural baseline (7-day rolling history)
    mean_amt_7d_hist: float = Field(0.0, description="7-day historical mean amount for entity")
    std_amt_7d_hist: float = Field(0.0, ge=0, description="7-day historical std of amount")
    z_amt_vs_7d: Optional[float] = Field(None, description="Z-score of current amount vs 7d baseline")

    # Propagation / campaign signals
    fp_cnt_24h: float = Field(..., ge=0, description="Count of fingerprint activity in past 24h")
    fp_cnt_72h: float = Field(..., ge=0, description="Count of fingerprint activity in past 72h")
    fp_growth_ratio_24h_over_72h: Optional[float] = Field(
        None, ge=0, description="Propagation growth ratio (24h/72h activity)"
    )

    @field_validator("z_amt_vs_7d", "fp_growth_ratio_24h_over_72h", mode="before")
    @classmethod
    def allow_none_float(cls, v):
        return v  # Allow None (will be imputed to 0 during standardization)

    model_config = {"json_schema_extra": {
        "example": {
            "TransactionID": 2987654,
            "TransactionAmt": 450.00,
            "entity_key": "4321||101||150||226||12345",
            "fingerprint_key": "Windows||gmail.com||W",
            "cnt_1h": 3,
            "sum_amt_1h": 890.50,
            "cnt_24h": 8,
            "avg_amt_24h": 312.00,
            "mean_amt_7d_hist": 95.00,
            "std_amt_7d_hist": 42.50,
            "z_amt_vs_7d": 8.35,
            "fp_cnt_24h": 12,
            "fp_cnt_72h": 15,
            "fp_growth_ratio_24h_over_72h": 0.80,
        }
    }}


class RiskScore(BaseModel):
    """Response schema: calibrated fraud risk score + operational band."""
    TransactionID: Optional[int]
    raw_fraud_probability: float = Field(..., description="Raw model probability before calibration")
    calibrated_fraud_probability: float = Field(
        ..., description="Temperature-scaled probability (operationally trusted)"
    )
    risk_band: str = Field(..., description="Operational risk tier: critical/high/medium/low")
    recommended_action: str = Field(..., description="Recommended fraud operations action")
    score_cutoffs: Dict[str, float] = Field(
        ..., description="Score thresholds used for band assignment"
    )
    model_version: str = Field("torch_mlp_v1", description="Model version identifier")


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_length=1000)


class BatchRiskScore(BaseModel):
    results: List[RiskScore]
    scored_count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    temperature_scaling: Optional[float]
    model_path: str
    startup_time_s: float


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _request_to_df(tx: TransactionRequest) -> pd.DataFrame:
    """Convert a TransactionRequest to a one-row DataFrame with all required columns."""
    row: dict[str, Any] = {
        "TransactionAmt": tx.TransactionAmt,
        "entity_key": tx.entity_key,
        "fingerprint_key": tx.fingerprint_key,
        "cnt_1h": tx.cnt_1h,
        "sum_amt_1h": tx.sum_amt_1h,
        "cnt_24h": tx.cnt_24h,
        "avg_amt_24h": tx.avg_amt_24h,
        "mean_amt_7d_hist": tx.mean_amt_7d_hist,
        "std_amt_7d_hist": tx.std_amt_7d_hist,
        "z_amt_vs_7d": tx.z_amt_vs_7d if tx.z_amt_vs_7d is not None else 0.0,
        "fp_cnt_24h": tx.fp_cnt_24h,
        "fp_cnt_72h": tx.fp_cnt_72h,
        "fp_growth_ratio_24h_over_72h": (
            tx.fp_growth_ratio_24h_over_72h
            if tx.fp_growth_ratio_24h_over_72h is not None
            else 0.0
        ),
    }
    return pd.DataFrame([row])


def _assign_band(score: float) -> str:
    if score >= SCORE_CUTOFFS["critical"]:
        return "critical"
    if score >= SCORE_CUTOFFS["high"]:
        return "high"
    if score >= SCORE_CUTOFFS["medium"]:
        return "medium"
    return "low"


def _score_single(tx: TransactionRequest) -> RiskScore:
    if _artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Server is starting up.",
        )

    df = _request_to_df(tx)

    # Raw score (no temperature)
    raw_probs = predict_proba_torch(df, _artifact, temperature=None)
    raw_score = float(raw_probs[0])

    # Calibrated score
    cal_probs = predict_proba_torch(df, _artifact, temperature=_temperature)
    cal_score = float(cal_probs[0])

    band = _assign_band(cal_score)

    return RiskScore(
        TransactionID=tx.TransactionID,
        raw_fraud_probability=round(raw_score, 6),
        calibrated_fraud_probability=round(cal_score, 6),
        risk_band=band,
        recommended_action=BAND_ACTIONS[band],
        score_cutoffs=SCORE_CUTOFFS,
        model_version="torch_mlp_v1",
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health_check() -> HealthResponse:
    """
    Liveness + readiness check.

    Returns model load status, temperature scaling value,
    and startup time. Use this for Kubernetes readiness probes.
    """
    return HealthResponse(
        status="healthy" if _artifact is not None else "starting",
        model_loaded=_artifact is not None,
        temperature_scaling=_temperature,
        model_path=str(MODEL_PATH),
        startup_time_s=round(_startup_time, 3),
    )


@app.post(
    "/score",
    response_model=RiskScore,
    tags=["Scoring"],
    summary="Score a single transaction",
    response_description="Calibrated fraud probability and risk band",
)
def score_transaction(request: TransactionRequest) -> RiskScore:
    """
    Score a single transaction and return a calibrated fraud risk score.

    **Pipeline:**
    1. Feature validation via Pydantic
    2. Standardization (train-fit stats from artefact)
    3. Torch MLP forward pass
    4. Temperature scaling calibration
    5. Capacity-based risk band assignment

    **Risk Bands:**
    | Band     | Score Threshold | Recommended Action       |
    |----------|----------------|--------------------------|
    | critical | ≥ 0.953        | Block / Immediate Review |
    | high     | ≥ 0.912        | Step-Up Authentication   |
    | medium   | ≥ 0.743        | Monitor / Delay          |
    | low      | < 0.743        | Auto-Approve             |
    """
    try:
        return _score_single(request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scoring error for TransactionID=%s", request.TransactionID)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc


@app.post(
    "/score/batch",
    response_model=BatchRiskScore,
    tags=["Scoring"],
    summary="Score a batch of transactions (max 1000)",
)
def score_batch(request: BatchRequest) -> BatchRiskScore:
    """
    Score a batch of up to 1,000 transactions.

    Useful for near-real-time micro-batch processing or testing.
    For large-scale batch inference, use the CLI pipeline instead:

        make infer
    """
    if _artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Server is starting up.",
        )

    t0 = time.time()
    results: List[RiskScore] = []

    try:
        for tx in request.transactions:
            results.append(_score_single(tx))
    except Exception as exc:
        logger.exception("Batch scoring error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {exc}",
        ) from exc

    latency_ms = (time.time() - t0) * 1000

    return BatchRiskScore(
        results=results,
        scored_count=len(results),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/", tags=["Operations"])
def root():
    """API root — redirect to /docs for interactive documentation."""
    return {
        "service": "Proactive Scam Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["/score", "/score/batch"],
    }
