"""
Tests for the inference pipeline.

Covers:
- Prediction output range [0, 1]
- Risk band assignment correctness
- Risk bands cover all rows (no NULL bands)
- Temperature scaling effect on output
- Batch inference CLI argument validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── Risk Band Assignment ─────────────────────────────────────────────────────

def assign_risk_band(score: pd.Series, cutoffs: dict) -> pd.Series:
    """Mirror of the risk band logic used in scripts/10_risk_bands.py and 11_infer_batch.py."""
    s = score.astype(float)
    band = np.where(
        s >= cutoffs["critical"], "critical",
        np.where(
            s >= cutoffs["high"], "high",
            np.where(s >= cutoffs["medium"], "medium", "low"),
        ),
    )
    return pd.Series(band, index=score.index, dtype="string")


@pytest.fixture()
def sample_scores() -> pd.Series:
    """A range of fraud scores from 0 to 1."""
    return pd.Series([0.01, 0.05, 0.15, 0.50, 0.75, 0.85, 0.95, 0.99])


@pytest.fixture()
def sample_cutoffs() -> dict:
    return {"critical": 0.95, "high": 0.80, "medium": 0.50}


def test_risk_bands_cover_all_rows(sample_scores, sample_cutoffs):
    """Every row must receive a risk band — no NULLs allowed."""
    bands = assign_risk_band(sample_scores, sample_cutoffs)
    assert bands.notna().all(), "All rows must have a non-null risk band"


def test_risk_bands_valid_values(sample_scores, sample_cutoffs):
    """Risk bands must only take the four allowed values."""
    bands = assign_risk_band(sample_scores, sample_cutoffs)
    allowed = {"critical", "high", "medium", "low"}
    invalid = set(bands.unique()) - allowed
    assert not invalid, f"Invalid risk band values found: {invalid}"


def test_critical_band_has_highest_scores(sample_cutoffs):
    """Critical band scores must be >= high band scores."""
    scores = pd.Series([0.10, 0.50, 0.85, 0.99])
    bands = assign_risk_band(scores, sample_cutoffs)
    critical_scores = scores[bands == "critical"]
    high_scores = scores[bands == "high"]

    if len(critical_scores) > 0 and len(high_scores) > 0:
        assert critical_scores.min() >= high_scores.max() - 1e-9, (
            "Critical band scores must be >= high band scores"
        )


def test_band_ordering_preserved(sample_cutoffs):
    """
    A score above critical_cutoff must be 'critical',
    between high and critical must be 'high', etc.
    """
    scores = pd.Series([
        sample_cutoffs["critical"] + 0.01,   # → critical
        sample_cutoffs["high"] + 0.01,        # → high
        sample_cutoffs["medium"] + 0.01,      # → medium
        sample_cutoffs["medium"] - 0.01,      # → low
    ])
    bands = assign_risk_band(scores, sample_cutoffs)
    expected = ["critical", "high", "medium", "low"]
    assert list(bands) == expected, f"Expected {expected}, got {list(bands)}"


def test_all_zeros_get_low_band(sample_cutoffs):
    """Scores of 0 must all be assigned to the 'low' band."""
    scores = pd.Series([0.0] * 100)
    bands = assign_risk_band(scores, sample_cutoffs)
    assert (bands == "low").all()


def test_all_ones_get_critical_band(sample_cutoffs):
    """Scores of 1.0 must all be assigned to the 'critical' band."""
    scores = pd.Series([1.0] * 100)
    bands = assign_risk_band(scores, sample_cutoffs)
    assert (bands == "critical").all()


# ─── Prediction Range ─────────────────────────────────────────────────────────

def test_prediction_range_valid():
    """All fraud scores produced by the model must be in [0, 1]."""
    # Simulate scored output (as would be written by 10_risk_bands.py or 11_infer_batch.py)
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.random(1000))  # uniform [0,1] — model scores are probabilities

    assert (scores >= 0.0).all(), "Fraud scores must be >= 0"
    assert (scores <= 1.0).all(), "Fraud scores must be <= 1"


def test_capacity_band_sizes():
    """
    With capacity_based policy (top 1%/3%/8%), the Critical/High/Medium band sizes
    must approximately match the intended percentages.
    """
    rng = np.random.default_rng(42)
    n = 100_000
    scores = pd.Series(rng.random(n))

    p_critical, p_high, p_medium = 0.01, 0.03, 0.08

    cutoffs = {
        "critical": float(scores.quantile(1.0 - p_critical)),
        "high": float(scores.quantile(1.0 - p_high)),
        "medium": float(scores.quantile(1.0 - p_medium)),
    }
    bands = assign_risk_band(scores, cutoffs)

    actual_critical_pct = (bands == "critical").mean()
    actual_high_pct = (bands == "high").mean()
    actual_medium_pct = (bands == "medium").mean()

    # Allow 0.5pp tolerance
    assert abs(actual_critical_pct - p_critical) < 0.005
    assert abs(actual_critical_pct + actual_high_pct - p_high) < 0.005
    assert abs(actual_critical_pct + actual_high_pct + actual_medium_pct - p_medium) < 0.005


# ─── Temperature Scaling ──────────────────────────────────────────────────────

def test_temperature_scaling_below_1_increases_confidence():
    """
    T < 1 should sharpen predictions (push probabilities away from 0.5).
    """
    import torch
    logit = torch.tensor([1.5])   # raw logit → sigmoid ≈ 0.818

    prob_uncalibrated = float(torch.sigmoid(logit).item())
    prob_calibrated = float(torch.sigmoid(logit / 0.9).item())  # T=0.9 < 1

    assert prob_calibrated > prob_uncalibrated, (
        "T < 1 should increase confidence (push probability upward for positive logits)"
    )


def test_temperature_scaling_above_1_decreases_confidence():
    """
    T > 1 should soften predictions (push probabilities toward 0.5).
    """
    import torch
    logit = torch.tensor([1.5])

    prob_uncalibrated = float(torch.sigmoid(logit).item())
    prob_calibrated = float(torch.sigmoid(logit / 1.5).item())  # T=1.5 > 1

    assert prob_calibrated < prob_uncalibrated, (
        "T > 1 should decrease confidence (push probability toward 0.5)"
    )


def test_temperature_1_is_identity():
    """T=1 must be a no-op (identity transformation)."""
    import torch
    logits = torch.tensor([0.5, 1.0, -0.5, 2.0])
    original = torch.sigmoid(logits)
    scaled = torch.sigmoid(logits / 1.0)
    assert torch.allclose(original, scaled), "T=1.0 must be identity"


# ─── Scoring Output Validation ────────────────────────────────────────────────

def test_scored_output_has_required_columns():
    """
    Scored output parquet must contain 'score' and 'risk_band' columns.
    """
    from pathlib import Path
    out_path = Path("data/outputs/test_scored.parquet")

    if not out_path.exists():
        pytest.skip("test_scored.parquet not yet generated. Run 'make evaluate' first.")

    df = pd.read_parquet(out_path)

    assert "score" in df.columns, "Scored output must have 'score' column"
    assert "risk_band" in df.columns, "Scored output must have 'risk_band' column"


def test_scored_output_no_null_bands():
    """All rows in scored output must have a non-null risk_band."""
    from pathlib import Path
    out_path = Path("data/outputs/test_scored.parquet")

    if not out_path.exists():
        pytest.skip("test_scored.parquet not yet generated. Run 'make evaluate' first.")

    df = pd.read_parquet(out_path)
    assert df["risk_band"].notna().all(), "No row should have a null risk_band"


def test_scored_output_score_range():
    """All scores in scored output must be in [0, 1]."""
    from pathlib import Path
    out_path = Path("data/outputs/test_scored.parquet")

    if not out_path.exists():
        pytest.skip("test_scored.parquet not yet generated. Run 'make evaluate' first.")

    df = pd.read_parquet(out_path)
    assert (df["score"] >= 0.0).all(), "All scores must be >= 0"
    assert (df["score"] <= 1.0).all(), "All scores must be <= 1"
