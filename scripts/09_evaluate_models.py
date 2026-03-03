"""

==============================
Stage 9 — Head-to-head model evaluation, SHAP explanations, and MLflow logging.

Compares the Logistic Regression baseline against the calibrated Torch MLP on
the held-out temporal test set across three complementary lenses:

  1. Standard ML metrics   — AUC-ROC, AUC-PR, Brier Score
  2. Operational lift      — precision, recall, and lift at top 1%, 3%, 8%
  3. Calibration           — reliability diagrams (quantile-binned)
  4. Explainability        — SHAP global feature importance (LR numeric features)

All metrics and generated artefacts (JSON, PNG) are logged to MLflow so every
evaluation run is fully reproducible and comparable.

Inputs
------
data/processed/dataset_v1.parquet
models/baseline_lr.joblib
models/torch_mlp.pt
models/torch_temperature.json  (optional — uncalibrated if absent)

Outputs
-------
reports/model_comparison.json                  — full metric comparison dict
reports/pr_curves.png                          — precision-recall curves
reports/calibration_curves.png                 — reliability diagrams
reports/figures/shap_global_importance.png     — SHAP bar chart
reports/shap_feature_importance.json           — machine-readable SHAP values
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.config import ProjectPaths
from src.evaluation.metrics import best_f1_threshold, binary_metrics
from src.inference.torch_infer import load_torch_artifact, predict_proba_torch
from src.utils.json_utils import save_json
from src.utils.logging import setup_logger


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pr_curves(y_true: np.ndarray, probs: dict[str, np.ndarray], out_path: Path) -> None:
    from sklearn.metrics import precision_recall_curve

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"baseline_lr": "#e74c3c", "torch_mlp_calibrated": "#2980b9"}
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=name, color=colors.get(name, "gray"), linewidth=2)

    base_rate = y_true.mean()
    ax.axhline(
        base_rate, linestyle="--", color="#7f8c8d", linewidth=1,
        label=f"Base rate ({base_rate:.3f})",
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curves (Temporal Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(
    y_true: np.ndarray, probs: dict[str, np.ndarray], out_path: Path, n_bins: int = 10
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"baseline_lr": "#e74c3c", "torch_mlp_calibrated": "#2980b9"}
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(
            mean_pred, frac_pos, marker="o", linewidth=2,
            label=name, color=colors.get(name, "gray"),
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#2ecc71", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curves (Quantile Binning)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_lift_at_k(
    y_true: np.ndarray, y_prob: np.ndarray, top_pcts: list[float]
) -> dict[str, float]:
    """
    Compute precision, recall, and lift at the top K% of scored transactions.
    Lift = precision_at_k / base_rate (how many times better than random review).
    """
    base_rate = y_true.mean()
    n = len(y_true)
    sorted_idx = np.argsort(y_prob)[::-1]
    results: dict[str, float] = {}

    for pct in top_pcts:
        k = max(1, int(n * pct / 100))
        top_k_labels = y_true[sorted_idx[:k]]
        prec_at_k = float(top_k_labels.mean())
        recall_at_k = float(top_k_labels.sum() / max(y_true.sum(), 1))
        lift = prec_at_k / max(base_rate, 1e-12)
        results[f"top_{pct}pct_precision"] = prec_at_k
        results[f"top_{pct}pct_recall"] = recall_at_k
        results[f"top_{pct}pct_lift"] = float(lift)

    return results


# ─── SHAP Explanations ────────────────────────────────────────────────────────

def compute_shap_explanations(
    lr_model,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    out_dir: Path,
    logger,
    n_background: int = 500,
    n_explain: int = 2000,
) -> None:
    """
    Compute SHAP values for the baseline LR model.

    The LR pipeline contains a FeatureHasher so its coef_ lives in a very
    high-dimensional space (n_features=2^18).  We must pass the **full**
    pre-processed feature matrix to LinearExplainer, then slice out the SHAP
    contributions that correspond to the numeric columns for plotting.

    Outputs:
      - reports/figures/shap_global_importance.png  (bar chart of mean |SHAP|)
      - reports/shap_feature_importance.json        (machine-readable importance)
    """
    try:
        import shap
        import scipy.sparse
    except ImportError:
        logger.warning("shap not installed. Skipping SHAP. Run: pip install shap")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols_available = [c for c in numeric_cols if c in X_test.columns]
    if not num_cols_available:
        logger.warning("No numeric columns available for SHAP. Skipping.")
        return

    logger.info("Computing SHAP on %d samples (full pipeline feature space)", n_explain)

    pre = lr_model.named_steps["pre"]
    clf = lr_model.named_steps["model"]

    # Transform the FULL feature set so dimensionality matches clf.coef_
    X_sub_all = X_test[feature_cols].copy().head(n_explain)
    X_bg_all  = X_test[feature_cols].copy().head(n_background)

    try:
        X_sub_t = pre.transform(X_sub_all)
        X_bg_t  = pre.transform(X_bg_all)
    except Exception as e:
        logger.warning("Could not transform features for SHAP: %s. Skipping.", e)
        return

    try:
        explainer = shap.LinearExplainer(clf, X_bg_t)
        shap_values_full = explainer.shap_values(X_sub_t)
    except Exception as e:
        logger.warning("SHAP LinearExplainer failed: %s. Skipping.", e)
        return

    # Identify which output columns belong to the numeric (StandardScaler) step.
    # ColumnTransformer names them "num__<original_col>" in get_feature_names_out().
    try:
        feature_names_out = pre.get_feature_names_out()
        num_indices    = [i for i, n in enumerate(feature_names_out) if n.startswith("num__")]
        display_names  = [n.replace("num__", "") for n in feature_names_out if n.startswith("num__")]
    except Exception:
        # Fallback: numeric transformer is first → first n_num columns
        n_num         = len(num_cols_available)
        num_indices   = list(range(n_num))
        display_names = num_cols_available

    if not num_indices:
        logger.warning("Could not locate numeric feature indices in transformer output. Skipping.")
        return

    # Slice to numeric columns only (handle both dense and sparse)
    if scipy.sparse.issparse(shap_values_full):
        shap_numeric = np.asarray(shap_values_full[:, num_indices].todense())
    else:
        shap_numeric = np.asarray(shap_values_full)[:, num_indices]

    # Global feature importance bar chart
    mean_abs_shap = np.abs(shap_numeric).mean(axis=0)
    feat_imp = sorted(zip(display_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    names, vals = zip(*feat_imp)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(names)[::-1], list(vals)[::-1], color="#2980b9", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value| — impact on fraud probability", fontsize=11)
    ax.set_title(
        "Feature Importance via SHAP\n(Baseline Logistic Regression, Numeric Features)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    global_imp_path = out_dir / "shap_global_importance.png"
    fig.savefig(global_imp_path, dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP importance chart: %s", global_imp_path)

    importance_json: dict = {
        "method": "SHAP LinearExplainer",
        "model": "LogisticRegression (numeric SHAP slice from full pipeline)",
        "n_samples_explained": n_explain,
        "feature_importance": [
            {"feature": name, "mean_abs_shap": float(val)}
            for name, val in zip(names, vals)
        ],
    }
    imp_path = out_dir.parent / "shap_feature_importance.json"
    with imp_path.open("w") as f:
        json.dump(importance_json, f, indent=2)
    logger.info("Saved SHAP importance JSON: %s", imp_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Step 9: Evaluate baseline LR vs Torch MLP, compute lift metrics, SHAP, log to MLflow.

    Inputs:
        data/processed/dataset_v1.parquet
        models/baseline_lr.joblib
        models/torch_mlp.pt
        (optional) models/torch_temperature.json

    Outputs:
        reports/model_comparison.json
        reports/pr_curves.png
        reports/calibration_curves.png
        reports/figures/shap_global_importance.png
        reports/shap_feature_importance.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    data_path = paths.data_processed / "dataset_v1.parquet"
    lr_path = Path("models/baseline_lr.joblib")
    torch_path = Path("models/torch_mlp.pt")
    temp_path = Path("models/torch_temperature.json")

    for p in [data_path, lr_path, torch_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    logger.info("Loading dataset: %s", data_path)
    df = pd.read_parquet(data_path)

    required = {"split", "y_ew_72h"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    y_valid = valid_df["y_ew_72h"].astype(int).to_numpy()
    y_test = test_df["y_ew_72h"].astype(int).to_numpy()

    feature_cols = [c for c in df.columns if c not in ["split", "y_ew_72h"]]
    numeric_cols = [
        "TransactionAmt", "cnt_1h", "sum_amt_1h", "cnt_24h", "avg_amt_24h",
        "mean_amt_7d_hist", "std_amt_7d_hist", "z_amt_vs_7d",
        "fp_cnt_24h", "fp_cnt_72h", "fp_growth_ratio_24h_over_72h",
    ]
    numeric_cols = [c for c in numeric_cols if c in test_df.columns]

    # ── Score models ──────────────────────────────────────────────────────────
    logger.info("Loading and scoring baseline LR")
    lr = joblib.load(lr_path)
    lr_valid = lr.predict_proba(valid_df[feature_cols])[:, 1]
    lr_test = lr.predict_proba(test_df[feature_cols])[:, 1]

    logger.info("Loading and scoring Torch MLP")
    torch_art = load_torch_artifact(torch_path)

    temperature = None
    if temp_path.exists():
        with temp_path.open() as f:
            temperature = float(json.load(f).get("temperature", 1.0))
        logger.info("Temperature scaling applied: T=%.4f", temperature)

    torch_valid = predict_proba_torch(valid_df, torch_art, temperature=temperature)
    torch_test = predict_proba_torch(test_df, torch_art, temperature=temperature)

    # ── Build report ──────────────────────────────────────────────────────────
    lift_metrics = compute_lift_at_k(y_test, torch_test, top_pcts=[1.0, 3.0, 8.0])

    report = {
        "dataset": {"rows_valid": int(len(valid_df)), "rows_test": int(len(test_df))},
        "base_rate": float(y_test.mean()),
        "models": {
            "baseline_lr": {
                "valid": {
                    "metrics": binary_metrics(y_valid, lr_valid),
                    "threshold": best_f1_threshold(y_valid, lr_valid),
                },
                "test": {"metrics": binary_metrics(y_test, lr_test)},
            },
            "torch_mlp_calibrated": {
                "valid": {
                    "metrics": binary_metrics(y_valid, torch_valid),
                    "threshold": best_f1_threshold(y_valid, torch_valid),
                },
                "test": {
                    "metrics": binary_metrics(y_test, torch_test),
                    "lift_at_k": lift_metrics,
                },
            },
        },
        "notes": (
            "Torch MLP is the primary model (temperature calibrated). "
            "Baseline LR provides an interpretable benchmark."
        ),
    }

    out_json = paths.reports / "model_comparison.json"
    out_pr = paths.reports / "pr_curves.png"
    out_cal = paths.reports / "calibration_curves.png"
    figures_dir = paths.reports / "figures"

    save_json(report, out_json)
    logger.info("Saved model comparison: %s", out_json)

    plot_pr_curves(
        y_test, {"baseline_lr": lr_test, "torch_mlp_calibrated": torch_test}, out_pr
    )
    logger.info("Saved PR curves: %s", out_pr)

    plot_calibration(
        y_test, {"baseline_lr": lr_test, "torch_mlp_calibrated": torch_test}, out_cal
    )
    logger.info("Saved calibration curves: %s", out_cal)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    logger.info("Computing SHAP explanations...")
    compute_shap_explanations(
        lr_model=lr,
        X_test=test_df[feature_cols],
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        out_dir=figures_dir,
        logger=logger,
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    torch_m = report["models"]["torch_mlp_calibrated"]["test"]["metrics"]
    lr_m = report["models"]["baseline_lr"]["test"]["metrics"]

    mlflow.set_experiment("scam-detection-evaluation")
    with mlflow.start_run(run_name="model-comparison-v1"):
        mlflow.log_metrics({
            "torch_test_auc_roc": torch_m["auc_roc"],
            "torch_test_avg_precision": torch_m["avg_precision"],
            "torch_test_brier": torch_m["brier"],
            "lr_test_auc_roc": lr_m["auc_roc"],
            "lr_test_avg_precision": lr_m["avg_precision"],
            "lr_test_brier": lr_m["brier"],
            **lift_metrics,
        })
        for artifact in [out_json, out_pr, out_cal]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        shap_imp_path = paths.reports / "shap_feature_importance.json"
        if shap_imp_path.exists():
            mlflow.log_artifact(str(shap_imp_path))

        shap_fig = figures_dir / "shap_global_importance.png"
        if shap_fig.exists():
            mlflow.log_artifact(str(shap_fig))

    logger.info(
        "Done. Torch test AP=%.4f AUC=%.4f | LR test AP=%.4f AUC=%.4f",
        torch_m["avg_precision"], torch_m["auc_roc"],
        lr_m["avg_precision"], lr_m["auc_roc"],
    )
    logger.info(
        "Operational lift at top-8%%: %.2f× | recall at top-8%%: %.1f%%",
        lift_metrics["top_8.0pct_lift"],
        lift_metrics["top_8.0pct_recall"] * 100,
    )


if __name__ == "__main__":
    main()
