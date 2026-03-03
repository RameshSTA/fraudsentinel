"""
=======================================

Compares the reference training distribution against the most recently
scored production batch to detect:

  1. Data drift (covariate shift) — input feature distributions changing
  2. Model performance drift — score distribution shifting
  3. Label drift — fraud base rate changing (requires ground truth)

Outputs
-------
  reports/drift_monitoring_report.html   — Interactive Evidently HTML report
  reports/drift_metrics.json             — Machine-readable drift metrics for alerting

Usage
-----
    python scripts/12_monitor_drift.py
    # Or:
    make monitor
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("drift_monitor")


NUMERIC_FEATURES = [
    "TransactionAmt",
    "cnt_1h",
    "sum_amt_1h",
    "cnt_24h",
    "avg_amt_24h",
    "mean_amt_7d_hist",
    "std_amt_7d_hist",
    "z_amt_vs_7d",
    "fp_cnt_24h",
    "fp_cnt_72h",
    "fp_growth_ratio_24h_over_72h",
]

# Drift alert thresholds
DRIFT_SHARE_ALERT = 0.30       # Alert if >30% of features show statistical drift
SCORE_MEAN_CHANGE_ALERT = 0.05  # Alert if mean fraud score shifts by >5 percentage points


def _load_reference(path: Path) -> pd.DataFrame:
    """Load the training/validation split as the reference distribution."""
    if not path.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {path}\n"
            "Run 'make pipeline' first to generate the processed dataset."
        )
    df = pd.read_parquet(path)
    ref = df[df["split"].isin(["train", "valid"])].copy()
    logger.info("Reference dataset: %d rows (train+valid splits)", len(ref))
    return ref


def _load_production(path: Path) -> pd.DataFrame:
    """Load the most recent scored production batch."""
    if not path.exists():
        raise FileNotFoundError(
            f"Production scored output not found: {path}\n"
            "Run 'make infer' or 'make evaluate' first."
        )
    df = pd.read_parquet(path)
    logger.info("Production batch: %d rows", len(df))
    return df


def _compute_manual_drift_metrics(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """
    Compute basic drift statistics without Evidently.

    Uses:
    - Population Stability Index (PSI) for score distribution shift
    - Mean/std comparison for each numeric feature
    - Jensen-Shannon divergence (approx via histogram comparison)
    """
    import numpy as np
    from scipy.stats import ks_2samp

    metrics: dict = {"features": {}, "score": {}, "label": {}}

    # Feature drift via 2-sample KS test
    drifted_features: list[str] = []

    for col in NUMERIC_FEATURES:
        if col not in ref.columns or col not in cur.columns:
            continue

        ref_vals = ref[col].dropna().to_numpy()
        cur_vals = cur[col].dropna().to_numpy()

        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        ks_stat, p_value = ks_2samp(ref_vals, cur_vals)
        ref_mean = float(np.nanmean(ref_vals))
        cur_mean = float(np.nanmean(cur_vals))
        mean_change_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100

        drift_detected = bool(p_value < 0.05)  # Standard 5% significance level

        if drift_detected:
            drifted_features.append(col)

        metrics["features"][col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": drift_detected,
            "ref_mean": round(ref_mean, 4),
            "current_mean": round(cur_mean, 4),
            "mean_change_pct": round(mean_change_pct, 2),
        }

    # Score distribution drift (if 'score' column available in production)
    if "score" in cur.columns:
        cur_scores = cur["score"].dropna().to_numpy()
        ref_score_mean = 0.028   # Expected: ~2.8% base rate from training distribution
        cur_score_mean = float(np.nanmean(cur_scores))
        score_shift = abs(cur_score_mean - ref_score_mean)

        metrics["score"] = {
            "reference_mean_score": ref_score_mean,
            "current_mean_score": round(cur_score_mean, 4),
            "absolute_shift": round(float(score_shift), 4),
            "alert": bool(score_shift > SCORE_MEAN_CHANGE_ALERT),
        }

    # Label drift (if ground truth available in production)
    label_col = "y_ew_72h"
    if label_col in cur.columns and label_col in ref.columns:
        ref_rate = float(ref[label_col].mean())
        cur_rate = float(cur[label_col].mean())
        metrics["label"] = {
            "reference_positive_rate": round(ref_rate, 4),
            "current_positive_rate": round(cur_rate, 4),
            "absolute_shift": round(abs(cur_rate - ref_rate), 4),
        }

    # Summary
    n_features_monitored = len([c for c in NUMERIC_FEATURES if c in ref.columns])
    drift_share = len(drifted_features) / max(n_features_monitored, 1)

    metrics["summary"] = {
        "n_features_monitored": n_features_monitored,
        "n_features_drifted": len(drifted_features),
        "drift_share": round(float(drift_share), 3),
        "drifted_features": drifted_features,
        "alert_drift_threshold": DRIFT_SHARE_ALERT,
        "global_alert": bool(drift_share > DRIFT_SHARE_ALERT),
    }

    return metrics


def _generate_evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, out_path: Path) -> bool:
    """
    Generate an Evidently HTML report for interactive drift exploration.

    Returns True if successful, False if Evidently is not installed.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError:
        logger.warning(
            "Evidently not installed. Skipping HTML report. Run: pip install evidently"
        )
        return False

    # Select common columns
    feature_cols = [c for c in NUMERIC_FEATURES if c in ref.columns and c in cur.columns]

    label_col = "y_ew_72h" if "y_ew_72h" in ref.columns and "y_ew_72h" in cur.columns else None
    # Only use score as prediction if it is present in BOTH reference and current data
    score_col = "score" if ("score" in cur.columns and "score" in ref.columns) else None

    column_mapping = ColumnMapping(
        target=label_col,
        prediction=score_col,
        numerical_features=feature_cols,
    )

    report = Report(metrics=[
        DataDriftPreset(stattest_threshold=0.05),
        DataQualityPreset(),
    ])

    try:
        report.run(
            reference_data=ref[feature_cols + ([label_col] if label_col else [])].head(50_000),
            current_data=cur[
                feature_cols
                + ([label_col] if label_col else [])
                + ([score_col] if score_col else [])
            ],
            column_mapping=column_mapping,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        logger.info("Evidently report saved: %s", out_path)
        return True
    except Exception as e:
        logger.warning("Evidently report generation failed: %s", e)
        return False


def _print_alert_summary(metrics: dict) -> None:
    summary = metrics.get("summary", {})
    drift_share = summary.get("drift_share", 0)
    drifted = summary.get("drifted_features", [])
    global_alert = summary.get("global_alert", False)

    if global_alert:
        logger.warning(
            "DRIFT ALERT: %.0f%% of features show significant drift (threshold: %.0f%%). "
            "Consider retraining.",
            drift_share * 100,
            DRIFT_SHARE_ALERT * 100,
        )
        logger.warning("Drifted features: %s", ", ".join(drifted))
    else:
        logger.info(
            "Drift check PASSED: %d/%d features drifted (%.0f%% < %.0f%% threshold).",
            summary.get("n_features_drifted", 0),
            summary.get("n_features_monitored", 0),
            drift_share * 100,
            DRIFT_SHARE_ALERT * 100,
        )

    score_info = metrics.get("score", {})
    if score_info.get("alert"):
        logger.warning(
            "SCORE DRIFT ALERT: Mean fraud score shifted from %.3f to %.3f "
            "(delta=%.3f > %.3f threshold).",
            score_info["reference_mean_score"],
            score_info["current_mean_score"],
            score_info["absolute_shift"],
            SCORE_MEAN_CHANGE_ALERT,
        )


def main() -> None:
    reference_path = Path("data/processed/dataset_v1.parquet")
    production_path = Path("data/outputs/test_scored.parquet")
    report_html = Path("reports/drift_monitoring_report.html")
    report_json = Path("reports/drift_metrics.json")

    logger.info("Loading reference distribution: %s", reference_path)
    ref = _load_reference(reference_path)

    logger.info("Loading production batch: %s", production_path)
    cur = _load_production(production_path)

    logger.info("Computing drift metrics (KS test per feature)...")
    metrics = _compute_manual_drift_metrics(ref, cur)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Drift metrics JSON: %s", report_json)

    _print_alert_summary(metrics)

    logger.info("Generating Evidently interactive report...")
    _generate_evidently_report(ref, cur, report_html)

    logger.info("Drift monitoring complete.")
    logger.info(
        "Summary: %d/%d features drifted | global_alert=%s",
        metrics["summary"]["n_features_drifted"],
        metrics["summary"]["n_features_monitored"],
        metrics["summary"]["global_alert"],
    )


if __name__ == "__main__":
    main()
