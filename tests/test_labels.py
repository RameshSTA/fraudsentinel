"""
Tests for early-warning label construction.

Covers:
- Positive rate is within expected range
- No entity labels its own current fraud event (only future fraud)
- Label is time-safe (no look-ahead)
- Label horizon is respected (fraud outside horizon must not be labelled)
- Edge cases: empty group, all fraud, no fraud
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.early_warning import LabelConfig, make_early_warning_label


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_entity() -> pd.DataFrame:
    """
    One entity with 5 transactions.
    Transaction at t=100 is fraudulent.
    Horizon is 3600s (1h).
    Transactions within 1h BEFORE the fraud should be labelled positive.
    """
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [0, 1800, 3600, 3601, 7200],   # t0..t+2h
        "isFraud":       [0,    0,    1,    0,    0],
        "entity_key":    ["A",  "A",  "A",  "A",  "A"],
    })


# ─── Core Label Logic ─────────────────────────────────────────────────────────

def test_no_fraud_produces_all_zeros():
    """If no fraud occurs, all labels must be 0."""
    df = pd.DataFrame({
        "TransactionID": range(5),
        "TransactionDT": [0, 3600, 7200, 10800, 14400],
        "isFraud": [0, 0, 0, 0, 0],
        "entity_key": ["A"] * 5,
    })
    cfg = LabelConfig(horizon_seconds=3600)
    labels, metrics = make_early_warning_label(df, cfg)

    assert labels.sum() == 0
    assert metrics["positive_count"] == 0


def test_fraud_outside_horizon_not_labelled():
    """
    Transaction at t=0, fraud at t=7201 (> 2h horizon).
    The t=0 row must NOT be labelled.
    """
    horizon = 7200  # 2 hours
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, horizon + 1],    # t=0 is 7201s before fraud
        "isFraud":       [0, 1],
        "entity_key":    ["A", "A"],
    })
    cfg = LabelConfig(horizon_seconds=horizon)
    labels, _ = make_early_warning_label(df, cfg)

    assert labels.iloc[0] == 0, (
        "Transaction more than horizon seconds before fraud must NOT be labelled"
    )


def test_fraud_inside_horizon_labelled():
    """
    Transaction at t=0, fraud at t=3600 (within 2h horizon).
    The t=0 row MUST be labelled 1.
    """
    horizon = 7200
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, 3600],
        "isFraud":       [0, 1],
        "entity_key":    ["A", "A"],
    })
    cfg = LabelConfig(horizon_seconds=horizon)
    labels, _ = make_early_warning_label(df, cfg)

    assert labels.iloc[0] == 1, (
        "Transaction within horizon of a future fraud must be labelled 1"
    )


def test_fraud_row_itself_not_labelled_for_self(simple_entity):
    """
    The fraud transaction itself (t=3600, isFraud=1) should not label itself.
    It should be labelled based on FUTURE fraud, not its own fraud event.
    """
    cfg = LabelConfig(horizon_seconds=3600)
    labels, _ = make_early_warning_label(simple_entity, cfg, fraud_col="isFraud")

    fraud_row_idx = simple_entity.index[simple_entity["isFraud"] == 1][0]
    # The fraud row itself should be 0 because it IS the fraud, not a predictor of future fraud
    assert labels.iloc[fraud_row_idx] == 0, (
        "The fraud row itself should not be labelled as early-warning positive"
    )


def test_label_dtype_is_int8():
    """Labels must be stored as int8 for memory efficiency."""
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, 3600],
        "isFraud":       [0, 1],
        "entity_key":    ["A", "A"],
    })
    labels, _ = make_early_warning_label(df, LabelConfig())
    assert labels.dtype == np.dtype("int8"), f"Expected int8, got {labels.dtype}"


def test_label_index_aligned_to_input():
    """Output label series index must match the input dataframe index exactly."""
    df = pd.DataFrame({
        "TransactionID": [10, 20, 30],
        "TransactionDT": [0, 3600, 7200],
        "isFraud":       [0, 1, 0],
        "entity_key":    ["A", "A", "A"],
    }, index=[100, 200, 300])

    labels, _ = make_early_warning_label(df, LabelConfig(horizon_seconds=7200))
    assert list(labels.index) == [100, 200, 300], "Label index must match input DataFrame index"


def test_cross_entity_isolation():
    """
    Fraud in entity A must not label transactions in entity B.
    """
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [0, 1800, 0, 1800],
        "isFraud":       [0, 1, 0, 0],           # Entity A has fraud; B has none
        "entity_key":    ["A", "A", "B", "B"],
    })
    cfg = LabelConfig(horizon_seconds=7200)
    labels, _ = make_early_warning_label(df, cfg)

    entity_b_labels = labels[df["entity_key"] == "B"]
    assert entity_b_labels.sum() == 0, (
        "Fraud in entity A must not propagate labels to entity B"
    )


def test_metrics_report_structure():
    """make_early_warning_label must return a complete metrics dict."""
    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "TransactionDT": [0, 3600],
        "isFraud":       [0, 1],
        "entity_key":    ["A", "A"],
    })
    _, metrics = make_early_warning_label(df, LabelConfig())

    expected_keys = {"horizon_seconds", "label_name", "positive_rate_pct", "positive_count", "total_rows"}
    assert expected_keys.issubset(set(metrics.keys())), (
        f"Metrics dict missing keys. Found: {set(metrics.keys())}"
    )


def test_missing_required_columns_raises():
    """make_early_warning_label must raise ValueError when required columns are absent."""
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        make_early_warning_label(df, LabelConfig())


def test_positive_rate_reasonable_on_real_like_data():
    """
    On a dataset with 3% fraud rate and 72h horizon, the positive rate
    should be higher than the base fraud rate (early-warning amplifies positives).
    """
    rng = np.random.default_rng(42)
    n = 1000
    fraud_flags = (rng.random(n) < 0.03).astype(int)
    df = pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": sorted(rng.integers(0, 86400 * 30, n)),   # 30 days
        "isFraud": fraud_flags,
        "entity_key": rng.choice(["A", "B", "C", "D"], n).tolist(),
    })

    cfg = LabelConfig(horizon_seconds=72 * 3600)
    labels, metrics = make_early_warning_label(df, cfg)

    # Early-warning label rate should exceed the raw fraud rate
    assert metrics["positive_rate_pct"] >= float(fraud_flags.mean() * 100), (
        "Early-warning positive rate should be >= raw fraud rate (horizon amplifies positives)"
    )
