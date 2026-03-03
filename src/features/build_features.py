from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """
    Parameters controlling feature computation windows (seconds).
    """
    win_1h: int = 3600
    win_24h: int = 86400
    win_72h: int = 3 * 86400
    win_7d: int = 7 * 86400


def make_entity_key(df: pd.DataFrame) -> pd.Series:
    """
    Create an entity proxy key for behavioural aggregation.

    We use a small set of fields often used as a stable proxy for an account/card/address entity.
    Missing components are kept as empty strings (missingness can be informative).

    Expected columns (when present):
    - card1, card2, card3, card5, addr1

    Returns
    -------
    pd.Series
        A string key, e.g. "card1||card2||card3||card5||addr1".
    """
    cols = ["card1", "card2", "card3", "card5", "addr1"]
    parts: list[pd.Series] = []
    for c in cols:
        if c in df.columns:
            parts.append(df[c].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), dtype="string"))
    return parts[0] + "||" + parts[1] + "||" + parts[2] + "||" + parts[3] + "||" + parts[4]


def make_fingerprint_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a propagation/campaign fingerprint key.

    This key approximates shared infrastructure used by scam/fraud campaigns.

    Expected columns (when present):
    - DeviceInfo, P_emaildomain, ProductCD

    Returns
    -------
    pd.Series
        A string key, e.g. "DeviceInfo||P_emaildomain||ProductCD".
    """
    def col(name: str) -> pd.Series:
        return df[name].astype("string").fillna("") if name in df.columns else pd.Series([""] * len(df), dtype="string")

    return col("DeviceInfo") + "||" + col("P_emaildomain") + "||" + col("ProductCD")


def add_entity_time_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Add behavioural velocity and history features per entity_key.

    Features:
    - cnt_1h, sum_amt_1h
    - cnt_24h, avg_amt_24h
    - mean_amt_7d_hist, std_amt_7d_hist (shifted; excludes current row)
    - z_amt_vs_7d

    Notes
    -----
    Uses time-based rolling windows by converting TransactionDT (seconds) into a datetime index.
    """
    out = df.sort_values(["entity_key", "TransactionDT"], kind="mergesort").copy()

    origin = pd.Timestamp("2000-01-01")
    out["_ts"] = origin + pd.to_timedelta(out["TransactionDT"].astype("int64"), unit="s")

    results: list[pd.DataFrame] = []

    for _, g in out.groupby("entity_key", sort=False):
        g = g.copy().set_index("_ts")

        # 1h window includes current row
        g["cnt_1h"] = g["TransactionAmt"].rolling(f"{cfg.win_1h}s").count().astype("int32")
        g["sum_amt_1h"] = g["TransactionAmt"].rolling(f"{cfg.win_1h}s").sum().fillna(0.0)

        # 24h window
        g["cnt_24h"] = g["TransactionAmt"].rolling(f"{cfg.win_24h}s").count().astype("int32")
        g["avg_amt_24h"] = g["TransactionAmt"].rolling(f"{cfg.win_24h}s").mean().fillna(0.0)

        # 7d historical window excluding current row
        amt_shift = g["TransactionAmt"].shift(1)
        g["mean_amt_7d_hist"] = amt_shift.rolling(f"{cfg.win_7d}s").mean().fillna(0.0)
        g["std_amt_7d_hist"] = amt_shift.rolling(f"{cfg.win_7d}s").std(ddof=0).fillna(0.0)

        denom = g["std_amt_7d_hist"].replace(0.0, np.nan)
        g["z_amt_vs_7d"] = (g["TransactionAmt"] - g["mean_amt_7d_hist"]) / denom

        results.append(g.reset_index(drop=True))

    out2 = pd.concat(results, ignore_index=True)
    return out2.drop(columns=["_ts"], errors="ignore")


def add_fingerprint_time_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Add propagation/campaign growth features per fingerprint_key.

    Features:
    - fp_cnt_24h: rolling count within 24h
    - fp_cnt_72h: rolling count within 72h
    - fp_growth_ratio_24h_over_72h: fp_cnt_24h / fp_cnt_72h (safe division)

    Notes
    -----
    This is a lightweight proxy for "scam propagation velocity".
    """
    out = df.sort_values(["fingerprint_key", "TransactionDT"], kind="mergesort").copy()

    origin = pd.Timestamp("2000-01-01")
    out["_ts"] = origin + pd.to_timedelta(out["TransactionDT"].astype("int64"), unit="s")

    results: list[pd.DataFrame] = []

    for _, g in out.groupby("fingerprint_key", sort=False):
        g = g.copy().set_index("_ts")

        g["fp_cnt_24h"] = g["TransactionID"].rolling(f"{cfg.win_24h}s").count().astype("int32")
        g["fp_cnt_72h"] = g["TransactionID"].rolling(f"{cfg.win_72h}s").count().astype("int32")

        denom = g["fp_cnt_72h"].replace(0, np.nan)
        g["fp_growth_ratio_24h_over_72h"] = (g["fp_cnt_24h"] / denom).astype("float32")

        results.append(g.reset_index(drop=True))

    out2 = pd.concat(results, ignore_index=True)
    return out2.drop(columns=["_ts"], errors="ignore")


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Orchestrate feature engineering (no labels, no splitting).

    Returns
    -------
    (df_features, report)
        df_features contains original columns + engineered features.
        report contains summary metrics to verify feature creation.
    """
    work = df.copy()

    # Key creation
    work["entity_key"] = make_entity_key(work)
    work["fingerprint_key"] = make_fingerprint_key(work)

    # Keep core columns + some identifiers. This avoids carrying 400+ columns while prototyping.
    keep_cols = [
        "TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
        "entity_key", "fingerprint_key",
        "ProductCD", "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo",
    ]
    keep_cols = [c for c in keep_cols if c in work.columns]
    work = work[keep_cols]

    # Required columns check
    for c in ("TransactionID", "TransactionDT", "TransactionAmt"):
        if c not in work.columns:
            raise ValueError(f"Missing required column for features: {c}")

    # Rolling features
    work = add_entity_time_features(work, cfg)
    work = add_fingerprint_time_features(work, cfg)

    # Report
    report: dict[str, Any] = {
        "rows": int(len(work)),
        "cols": int(work.shape[1]),
        "engineered_columns_present": [c for c in [
            "cnt_1h", "sum_amt_1h", "cnt_24h", "avg_amt_24h",
            "mean_amt_7d_hist", "std_amt_7d_hist", "z_amt_vs_7d",
            "fp_cnt_24h", "fp_cnt_72h", "fp_growth_ratio_24h_over_72h"
        ] if c in work.columns],
        "missingness_pct_engineered": {
            c: float(100.0 * work[c].isna().mean())
            for c in [
                "cnt_1h", "sum_amt_1h", "cnt_24h", "avg_amt_24h",
                "mean_amt_7d_hist", "std_amt_7d_hist", "z_amt_vs_7d",
                "fp_cnt_24h", "fp_cnt_72h", "fp_growth_ratio_24h_over_72h"
            ] if c in work.columns
        }
    }

    return work, report