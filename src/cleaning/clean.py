from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    """
    Configuration for cleaning rules.

    Notes
    -----
    For fraud/scam datasets, we intentionally avoid heavy imputation because
    missingness can be predictive. Cleaning focuses on type safety and
    minimal, reversible transformations.
    """
    enforce_transaction_id_unique: bool = True
    cast_types: bool = True


def clean_merged_dataset(df: pd.DataFrame, cfg: CleaningConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean the merged transaction + identity dataset with minimal, safe rules.

    Cleaning includes:
    - optional type casting for core columns
    - optional de-duplication on TransactionID
    - basic sanity checks and cleaning metrics

    Parameters
    ----------
    df:
        Merged dataframe produced by the load+merge step.
    cfg:
        Cleaning configuration.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Cleaned dataframe and a JSON-serializable metrics dict.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()

    metrics: dict[str, Any] = {
        "rows_before": int(len(out)),
        "cols_before": int(out.shape[1]),
    }

    # Optional: enforce unique TransactionID (keep first occurrence)
    if cfg.enforce_transaction_id_unique:
        dup_count = int(out.duplicated(subset=["TransactionID"]).sum())
        metrics["duplicate_transaction_id_rows"] = dup_count
        if dup_count > 0:
            out = out.drop_duplicates(subset=["TransactionID"], keep="first")

    # Type casting (minimal and safe)
    if cfg.cast_types:
        # TransactionDT should be integer seconds (nullable integer type)
        out["TransactionDT"] = pd.to_numeric(out["TransactionDT"], errors="coerce").astype("Int64")

        # Amount numeric
        out["TransactionAmt"] = pd.to_numeric(out["TransactionAmt"], errors="coerce")

        # Label as nullable integer 0/1
        out["isFraud"] = pd.to_numeric(out["isFraud"], errors="coerce").astype("Int64")

    # Metrics after
    metrics["rows_after"] = int(len(out))
    metrics["cols_after"] = int(out.shape[1])

    # Missingness of core fields after casting
    metrics["missing_core_pct"] = {
        "TransactionID": float(100.0 * out["TransactionID"].isna().mean()),
        "TransactionDT": float(100.0 * out["TransactionDT"].isna().mean()),
        "TransactionAmt": float(100.0 * out["TransactionAmt"].isna().mean()),
        "isFraud": float(100.0 * out["isFraud"].isna().mean()),
    }

    # Dtype snapshot for core columns
    metrics["core_dtypes"] = {
        "TransactionID": str(out["TransactionID"].dtype),
        "TransactionDT": str(out["TransactionDT"].dtype),
        "TransactionAmt": str(out["TransactionAmt"].dtype),
        "isFraud": str(out["isFraud"].dtype),
    }

    # Basic label stats
    y = out["isFraud"].dropna()
    metrics["label_counts"] = y.value_counts().to_dict()
    metrics["label_rate_pct"] = float(100.0 * y.mean()) if len(y) else None

    # TransactionDT sanity
    dt = out["TransactionDT"].dropna()
    metrics["transaction_dt_min"] = int(dt.min()) if len(dt) else None
    metrics["transaction_dt_max"] = int(dt.max()) if len(dt) else None

    # Amount sanity
    amt = out["TransactionAmt"].dropna()
    metrics["amount_summary"] = {
        "min": float(amt.min()) if len(amt) else None,
        "max": float(amt.max()) if len(amt) else None,
        "p50": float(amt.quantile(0.50)) if len(amt) else None,
        "p95": float(amt.quantile(0.95)) if len(amt) else None,
        "p99": float(amt.quantile(0.99)) if len(amt) else None,
    }

    # Identity coverage proxy (fields typically coming from identity table)
    identity_cols = [c for c in ("DeviceInfo", "DeviceType") if c in out.columns]
    if identity_cols:
        identity_any = out[identity_cols].notna().any(axis=1)
        metrics["identity_coverage_pct"] = float(100.0 * identity_any.mean())
        metrics["identity_columns_used"] = identity_cols
    else:
        metrics["identity_coverage_pct"] = None
        metrics["identity_columns_used"] = []

    # Top missingness columns (helps diagnose)
    missing_pct_all = (out.isna().mean() * 100.0).sort_values(ascending=False)
    metrics["top_missing_columns_pct"] = missing_pct_all.head(15).round(4).to_dict()

    return out, metrics