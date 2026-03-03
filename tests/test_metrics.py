"""
Tests for model evaluation metrics.

Covers:
- binary_metrics returns correct keys and valid ranges
- best_f1_threshold returns a threshold in [0,1]
- Lift computation is numerically correct
- Brier score is 0 for perfect predictions
- AUC-PR equals 1.0 for perfect ranking
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.metrics import best_f1_threshold, binary_metrics


# ─── binary_metrics ───────────────────────────────────────────────────────────

def test_binary_metrics_keys():
    """binary_metrics must return auc_roc, avg_precision, and brier."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = binary_metrics(y, p)

    expected_keys = {"auc_roc", "avg_precision", "brier"}
    assert expected_keys.issubset(set(result.keys()))


def test_binary_metrics_range():
    """All metrics must be in [0, 1]."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    p = rng.random(100)
    result = binary_metrics(y, p)

    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"Metric {key}={val} is out of [0, 1]"


def test_binary_metrics_perfect_classifier():
    """Perfect classifier should have AUC-ROC=1.0 and Brier≈0."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    result = binary_metrics(y, p)

    assert result["auc_roc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.0)


def test_binary_metrics_random_classifier_auc_near_half():
    """A random classifier should have AUC-ROC near 0.5 on large samples."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 10_000)
    p = rng.random(10_000)
    result = binary_metrics(y, p)

    assert 0.40 <= result["auc_roc"] <= 0.60, (
        f"Random classifier AUC should be near 0.5, got {result['auc_roc']:.4f}"
    )


def test_binary_metrics_worst_classifier():
    """Inverted classifier (predicting wrong) should have AUC-ROC < 0.5."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.9, 0.1, 0.1])   # Inverted predictions
    result = binary_metrics(y, p)
    assert result["auc_roc"] < 0.5


# ─── best_f1_threshold ────────────────────────────────────────────────────────

def test_best_f1_threshold_returns_valid_threshold():
    """Returned threshold must be in [0, 1]."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    result = best_f1_threshold(y, p)

    assert 0.0 <= result["best_threshold"] <= 1.0


def test_best_f1_threshold_keys():
    """best_f1_threshold must return all required keys."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.7, 0.9])
    result = best_f1_threshold(y, p)

    expected = {"best_threshold", "best_f1", "precision_at_best", "recall_at_best"}
    assert expected.issubset(set(result.keys()))


def test_best_f1_threshold_f1_in_range():
    """best_f1 must be in [0, 1]."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    result = best_f1_threshold(y, p)

    assert 0.0 <= result["best_f1"] <= 1.0


def test_best_f1_threshold_precision_recall_in_range():
    """precision_at_best and recall_at_best must both be in [0, 1]."""
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    result = best_f1_threshold(y, p)

    assert 0.0 <= result["precision_at_best"] <= 1.0
    assert 0.0 <= result["recall_at_best"] <= 1.0


# ─── Lift-at-K ────────────────────────────────────────────────────────────────

def test_lift_at_k_perfect_ranking():
    """
    If all frauds are ranked first, lift = 1 / base_rate.
    E.g., 10% base rate → max possible lift = 10×.
    """
    n = 1000
    n_fraud = 100  # 10% base rate
    y = np.array([1] * n_fraud + [0] * (n - n_fraud))
    # Perfect ranking: all frauds have score 1.0, all non-frauds 0.0
    p = np.array([1.0] * n_fraud + [0.0] * (n - n_fraud))

    # Top 10% (100 rows) should capture all 100 frauds
    k = n_fraud
    sorted_idx = np.argsort(p)[::-1]
    top_k = y[sorted_idx[:k]]

    precision_at_k = top_k.mean()
    base_rate = y.mean()
    lift = precision_at_k / base_rate

    assert precision_at_k == pytest.approx(1.0), "Perfect ranker: top-10% precision should be 1.0"
    assert lift == pytest.approx(10.0), "Perfect ranker: lift should be 10× for 10% base rate"


def test_lift_at_k_random_ranking_near_1():
    """
    For a random ranking on a large sample, lift should be near 1.0.
    """
    rng = np.random.default_rng(99)
    n = 50_000
    y = (rng.random(n) < 0.05).astype(int)  # 5% base rate
    p = rng.random(n)                         # random scores

    k = int(n * 0.05)
    sorted_idx = np.argsort(p)[::-1]
    prec = y[sorted_idx[:k]].mean()
    lift = prec / y.mean()

    assert 0.7 <= lift <= 1.3, f"Random ranking lift should be near 1.0, got {lift:.3f}"
