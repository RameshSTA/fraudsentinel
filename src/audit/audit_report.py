from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any

import pandas as pd


@dataclass(frozen=True)
class AuditConfig:
    """
    Configuration for audit reporting.

    Parameters
    ----------
    key_columns:
        Columns to include in missingness reporting when present.
    top_n:
        Top N most common values to show for selected categorical columns.
    """
    key_columns: tuple[str, ...] = (
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "isFraud",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
    )
    top_n: int = 10


def _safe_mean(series: pd.Series) -> float | None:
    """
    Return mean as float where possible, otherwise None.
    """
    try:
        return float(series.mean())
    except Exception:
        return None


def build_audit_report(df: pd.DataFrame, cfg: AuditConfig) -> dict[str, Any]:
    """
    Build a structured audit report for a dataset.

    This function does not modify the data. It only computes descriptive statistics
    to help validate ingestion/merge quality and understand missingness/imbalance.

    Parameters
    ----------
    df:
        Input dataframe to audit.
    cfg:
        AuditConfig describing which columns to focus on.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary containing summary metrics.
    """
    report: dict[str, Any] = {}

    # Basic shape
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    # Column presence
    present_cols = [c for c in cfg.key_columns if c in df.columns]
    missing_cols = [c for c in cfg.key_columns if c not in df.columns]
    report["columns"] = {"present_key_columns": present_cols, "missing_key_columns": missing_cols}

    # Time coverage
    if "TransactionDT" in df.columns:
        dt = pd.to_numeric(df["TransactionDT"], errors="coerce")
        report["time"] = {
            "transaction_dt_min": int(dt.min()) if pd.notna(dt.min()) else None,
            "transaction_dt_max": int(dt.max()) if pd.notna(dt.max()) else None,
            "transaction_dt_missing_pct": float(100.0 * dt.isna().mean()),
        }
    else:
        report["time"] = None

    # Label distribution
    if "isFraud" in df.columns:
        y = pd.to_numeric(df["isFraud"], errors="coerce")
        fraud_rate = _safe_mean(y)
        report["label"] = {
            "isFraud_missing_pct": float(100.0 * y.isna().mean()),
            "isFraud_rate_pct": float(100.0 * fraud_rate) if fraud_rate is not None else None,
            "isFraud_counts": df["isFraud"].value_counts(dropna=False).to_dict(),
        }
    else:
        report["label"] = None

    # Missingness for key columns
    miss: dict[str, float] = {}
    for c in present_cols:
        miss[c] = float(100.0 * df[c].isna().mean())
    report["missingness_pct"] = miss

    # Identity-join coverage proxy: presence of DeviceInfo/DeviceType (common in identity file)
    identity_cols = [c for c in ("DeviceInfo", "DeviceType") if c in df.columns]
    if identity_cols:
        identity_any = df[identity_cols].notna().any(axis=1)
        report["identity_coverage"] = {
            "columns_used": identity_cols,
            "rows_with_any_identity_pct": float(100.0 * identity_any.mean()),
        }
    else:
        report["identity_coverage"] = None

    # Quick numeric sanity checks
    numeric_checks: dict[str, Any] = {}
    for c in ("TransactionAmt", "card1", "card2", "addr1", "addr2"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            numeric_checks[c] = {
                "missing_pct": float(100.0 * s.isna().mean()),
                "min": float(s.min()) if pd.notna(s.min()) else None,
                "max": float(s.max()) if pd.notna(s.max()) else None,
                "p50": float(s.quantile(0.50)) if s.notna().any() else None,
                "p95": float(s.quantile(0.95)) if s.notna().any() else None,
                "p99": float(s.quantile(0.99)) if s.notna().any() else None,
            }
    report["numeric_sanity"] = numeric_checks

    # Top categories (small selection)
    cat_cols = [c for c in ("ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType") if c in df.columns]
    top_categories: dict[str, Any] = {}
    for c in cat_cols:
        vc = df[c].astype("string").value_counts(dropna=False).head(cfg.top_n)
        top_categories[c] = {str(k): int(v) for k, v in vc.items()}
    report["top_categories"] = top_categories

    return report


def save_json(report: Mapping[str, Any], out_path: Path) -> None:
    """
    Save a JSON report to disk.

    Parameters
    ----------
    report:
        JSON-serializable mapping.
    out_path:
        Output file path.
    """
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)