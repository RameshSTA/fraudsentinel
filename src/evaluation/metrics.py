"""
src/evaluation/metrics.py
==========================
Standard evaluation metrics for imbalanced binary classification.

Design notes
------------
- AUC-PR (``avg_precision``) is the *primary* metric for this problem.  On
  datasets with 2–3% fraud rate, AUC-ROC can be misleadingly high even for
  poor models because the large true-negative mass inflates the ROC curve.
  AUC-PR focuses exclusively on the precision/recall trade-off on the
  positive (fraud) class.

- Brier Score measures probabilistic calibration.  A perfectly calibrated
  model whose outputs are p̂ = 0.80 wherever the true fraud rate is 80% would
  achieve a Brier score equal to the base rate variance.  Lower is better.

- Best-F1 threshold is informational only.  Operationally, band thresholds
  are set by traffic-capacity percentiles (top 1%/3%/8%), not by F1.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """
    Compute the three core evaluation metrics for imbalanced binary classification.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).  Will be cast to ``int``.
    y_prob:
        Predicted probabilities for the positive class.  Must be in [0, 1].

    Returns
    -------
    dict with keys:
        ``auc_roc``       — Area under the ROC curve (macro, insensitive to imbalance).
        ``avg_precision`` — Area under the PR curve = weighted mean precision (primary metric).
        ``brier``         — Brier score; measures calibration quality (lower is better).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """
    Find the decision threshold that maximises the F1 score on the positive class.

    Useful for reporting and as a sanity check, but note that in this system
    operational thresholds are set by traffic-capacity percentiles rather than
    by F1 optimisation.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1).
    y_prob:
        Predicted probabilities for the positive class.

    Returns
    -------
    dict with keys:
        ``best_threshold``    — Score cutoff that maximises F1.
        ``best_f1``           — F1 score at that threshold.
        ``precision_at_best`` — Precision at the best-F1 threshold.
        ``recall_at_best``    — Recall at the best-F1 threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    p, r, thr = precision_recall_curve(y_true, y_prob)
    # Add epsilon to denominator to avoid divide-by-zero at p=r=0
    f1 = (2 * p * r) / (p + r + 1e-12)

    best_idx = int(np.nanargmax(f1))
    # thr has one fewer element than p and r; clamp index to valid range
    threshold = float(thr[max(best_idx - 1, 0)]) if thr.size else 0.5

    return {
        "best_threshold": threshold,
        "best_f1": float(f1[best_idx]),
        "precision_at_best": float(p[best_idx]),
        "recall_at_best": float(r[best_idx]),
    }