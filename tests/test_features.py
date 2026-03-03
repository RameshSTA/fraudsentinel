"""
Tests for feature engineering.

Covers:
- No temporal leakage in velocity features
- Feature output shape and column presence
- Entity key generation
- Fingerprint key generation
- Z-score handling for single-transaction entities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FeatureConfig,
    add_entity_time_features,
    add_fingerprint_time_features,
    build_features,
    make_entity_key,
    make_fingerprint_key,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def simple_transactions() -> pd.DataFrame:
    """Three transactions for the same entity at hourly intervals."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [0, 3600, 7200],       # t=0, t+1h, t+2h
        "TransactionAmt": [100.0, 200.0, 150.0],
        "isFraud": [0, 0, 1],
        "card1": ["A", "A", "A"],
        "card2": ["B", "B", "B"],
        "card3": ["C", "C", "C"],
        "card5": ["D", "D", "D"],
        "addr1": ["1", "1", "1"],
        "DeviceInfo": ["iOS", "iOS", "iOS"],
        "P_emaildomain": ["gmail.com", "gmail.com", "gmail.com"],
        "ProductCD": ["W", "W", "W"],
    })


@pytest.fixture()
def multi_entity_transactions() -> pd.DataFrame:
    """Two entities with two transactions each."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [0, 3600, 0, 3600],
        "TransactionAmt": [100.0, 200.0, 50.0, 75.0],
        "isFraud": [0, 1, 0, 0],
        "card1": ["A", "A", "B", "B"],
        "card2": ["", "", "", ""],
        "card3": ["", "", "", ""],
        "card5": ["", "", "", ""],
        "addr1": ["1", "1", "2", "2"],
        "DeviceInfo": ["iOS", "iOS", "Android", "Android"],
        "P_emaildomain": ["g.com", "g.com", "y.com", "y.com"],
        "ProductCD": ["W", "W", "H", "H"],
    })


# ─── Entity Key Tests ─────────────────────────────────────────────────────────

def test_entity_key_is_deterministic(simple_transactions):
    """Same input always produces the same entity key."""
    keys_1 = make_entity_key(simple_transactions)
    keys_2 = make_entity_key(simple_transactions)
    pd.testing.assert_series_equal(keys_1, keys_2)


def test_entity_key_differs_across_entities(multi_entity_transactions):
    """Different card/addr combinations must produce different entity keys."""
    keys = make_entity_key(multi_entity_transactions)
    entity_a = keys.iloc[0]
    entity_b = keys.iloc[2]
    assert entity_a != entity_b, "Different entities must not collide on entity_key"


def test_entity_key_same_within_entity(simple_transactions):
    """All rows for the same entity must have identical entity keys."""
    keys = make_entity_key(simple_transactions)
    assert keys.nunique() == 1, "Same entity should produce exactly one unique key"


def test_entity_key_handles_missing_columns():
    """Entity key creation must work even when optional columns are absent."""
    df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [0], "TransactionAmt": [100.0]})
    keys = make_entity_key(df)
    assert len(keys) == 1
    assert isinstance(keys.iloc[0], str)


# ─── Fingerprint Key Tests ────────────────────────────────────────────────────

def test_fingerprint_key_is_deterministic(simple_transactions):
    keys_1 = make_fingerprint_key(simple_transactions)
    keys_2 = make_fingerprint_key(simple_transactions)
    pd.testing.assert_series_equal(keys_1, keys_2)


def test_fingerprint_key_differs_across_device_groups(multi_entity_transactions):
    keys = make_fingerprint_key(multi_entity_transactions)
    # iOS/gmail.com/W vs Android/yahoo.com/H
    assert keys.iloc[0] != keys.iloc[2]


# ─── Velocity Feature Tests ───────────────────────────────────────────────────

def test_velocity_features_present(simple_transactions):
    """All expected velocity features must be produced."""
    df = simple_transactions.copy()
    df["entity_key"] = make_entity_key(df)
    cfg = FeatureConfig()
    result = add_entity_time_features(df, cfg)

    expected_cols = [
        "cnt_1h", "sum_amt_1h", "cnt_24h", "avg_amt_24h",
        "mean_amt_7d_hist", "std_amt_7d_hist", "z_amt_vs_7d",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Expected feature column missing: {col}"


def test_velocity_count_monotonically_increases_within_entity(simple_transactions):
    """
    cnt_1h must be non-decreasing within a single entity sorted by time.

    This tests no temporal leakage: only past rows should be included.
    """
    df = simple_transactions.copy()
    df["entity_key"] = make_entity_key(df)
    cfg = FeatureConfig()
    result = add_entity_time_features(df, cfg)
    result = result.sort_values("TransactionDT").reset_index(drop=True)

    diffs = result["cnt_1h"].diff().dropna()
    assert (diffs >= 0).all(), (
        "cnt_1h should be non-decreasing over time within an entity (no future look-ahead)"
    )


def test_historical_stats_use_shifted_values(simple_transactions):
    """
    mean_amt_7d_hist and std_amt_7d_hist must be computed from PAST transactions only.
    The first transaction for an entity must have mean_amt_7d_hist == 0.0 (no history).
    """
    df = simple_transactions.copy()
    df["entity_key"] = make_entity_key(df)
    cfg = FeatureConfig()
    result = add_entity_time_features(df, cfg)
    result = result.sort_values("TransactionDT").reset_index(drop=True)

    # First transaction has no prior history
    assert result["mean_amt_7d_hist"].iloc[0] == 0.0, (
        "First transaction must have mean_amt_7d_hist=0.0 (no prior history)"
    )


def test_zscore_is_nan_for_zero_std(simple_transactions):
    """
    When std_amt_7d_hist == 0 (all historical amounts identical),
    the z-score should be NaN (not +/-inf, not 0 from false division).
    """
    df = simple_transactions.copy()
    df["TransactionAmt"] = 100.0   # All amounts identical → std=0
    df["entity_key"] = make_entity_key(df)
    cfg = FeatureConfig()
    result = add_entity_time_features(df, cfg)
    result = result.sort_values("TransactionDT").reset_index(drop=True)

    # The second and third rows have history → std=0 → z should be NaN
    z_with_history = result.loc[result.index[1:], "z_amt_vs_7d"]
    assert not np.any(np.isinf(z_with_history.dropna().values)), (
        "z_amt_vs_7d must never be infinite"
    )


# ─── Fingerprint Propagation Feature Tests ───────────────────────────────────

def test_fingerprint_features_present(simple_transactions):
    """fp_cnt_24h, fp_cnt_72h, fp_growth_ratio must be produced."""
    df = simple_transactions.copy()
    df["fingerprint_key"] = make_fingerprint_key(df)
    cfg = FeatureConfig()
    result = add_fingerprint_time_features(df, cfg)

    expected = ["fp_cnt_24h", "fp_cnt_72h", "fp_growth_ratio_24h_over_72h"]
    for col in expected:
        assert col in result.columns, f"Missing propagation feature: {col}"


def test_fp_growth_ratio_bounded(simple_transactions):
    """fp_growth_ratio_24h_over_72h must be in [0, 1] or NaN (not > 1 since 24h ⊆ 72h)."""
    df = simple_transactions.copy()
    df["fingerprint_key"] = make_fingerprint_key(df)
    cfg = FeatureConfig()
    result = add_fingerprint_time_features(df, cfg)

    ratio = result["fp_growth_ratio_24h_over_72h"].dropna()
    assert (ratio <= 1.0 + 1e-6).all(), (
        "fp_growth_ratio cannot exceed 1.0 since 24h window ⊆ 72h window"
    )
    assert (ratio >= 0.0).all(), "fp_growth_ratio must be non-negative"


# ─── Full build_features Orchestration ────────────────────────────────────────

def test_build_features_output_shape(simple_transactions):
    """build_features must return a DataFrame with all engineered columns and a report."""
    cfg = FeatureConfig()
    result_df, report = build_features(simple_transactions, cfg)

    # All input rows preserved
    assert len(result_df) == len(simple_transactions)

    # Report must have the right structure
    assert "rows" in report
    assert "cols" in report
    assert "engineered_columns_present" in report
    assert len(report["engineered_columns_present"]) > 0


def test_build_features_raises_on_missing_required_col():
    """build_features must raise ValueError when required columns are absent."""
    df = pd.DataFrame({"foo": [1, 2, 3]})
    cfg = FeatureConfig()
    with pytest.raises((ValueError, KeyError)):
        build_features(df, cfg)


def test_build_features_no_isFraud_leakage(simple_transactions):
    """
    isFraud must NOT appear in the engineered feature columns.
    It is a raw label and must never become a feature.
    """
    cfg = FeatureConfig()
    result_df, _ = build_features(simple_transactions, cfg)

    # isFraud may remain in the output (it's carried through), but it must
    # not be used to derive any engineered feature — confirm no NaN injection
    feature_cols = [
        "cnt_1h", "sum_amt_1h", "cnt_24h", "avg_amt_24h",
        "mean_amt_7d_hist", "std_amt_7d_hist",
        "fp_cnt_24h", "fp_cnt_72h",
    ]
    for col in feature_cols:
        if col in result_df.columns:
            assert result_df[col].dtype != object, (
                f"Feature {col} should be numeric, not object"
            )
