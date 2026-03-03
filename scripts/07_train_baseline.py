"""
============================
Stage 7 — Train the interpretable Logistic Regression baseline.

Builds a full scikit-learn Pipeline that applies:
    - Median imputation + StandardScaler for numeric features
    - Mode imputation + OneHotEncoder for categorical features
    - FeatureHasher (2^18 buckets) for high-cardinality entity/fingerprint keys
    - LogisticRegression(solver='saga', class_weight='balanced')

The baseline serves two purposes:
  1. A strong, interpretable benchmark against which the Torch MLP is compared.
  2. Validation that carefully engineered features (not model complexity) drive
     the majority of the predictive signal.

All hyperparameters, metrics, and the fitted model artefact are logged to
MLflow for experiment tracking and reproducibility.

Input
-----
data/processed/dataset_v1.parquet

Outputs
-------
models/baseline_lr.joblib            — serialised sklearn Pipeline
reports/baseline_lr_metrics.json     — validation + test AUC-ROC, AUC-PR, F1 threshold
"""

from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.config import ProjectPaths
from src.modeling.baseline_pipeline import (
    BaselineConfig,
    best_f1_threshold,
    build_baseline_pipeline,
    evaluate_binary,
)
from src.utils.json_utils import save_json
from src.utils.logging import setup_logger


def main() -> None:
    """
    Step 7A: Train baseline sklearn model with MLflow experiment tracking.

    Input:
        data/processed/dataset_v1.parquet  (must include split + y_ew_72h)

    Outputs:
        models/baseline_lr.joblib
        reports/baseline_lr_metrics.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    in_path = paths.data_processed / "dataset_v1.parquet"
    model_path = Path("models/baseline_lr.joblib")
    report_path = paths.reports / "baseline_lr_metrics.json"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run Step 6 first.")

    logger.info("Loading processed dataset: %s", in_path)
    df = pd.read_parquet(in_path)

    cfg = BaselineConfig(target_col="y_ew_72h", split_col="split", hash_dim=2**18, random_state=42)

    required = {cfg.target_col, cfg.split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_df = df[df[cfg.split_col] == "train"].copy()
    valid_df = df[df[cfg.split_col] == "valid"].copy()
    test_df = df[df[cfg.split_col] == "test"].copy()

    logger.info("Split sizes: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df))

    pipe, cols_used = build_baseline_pipeline(df, cfg)

    # Feature columns are everything except target + split
    feature_cols = [c for c in df.columns if c not in [cfg.target_col, cfg.split_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[cfg.target_col].astype(int).values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[cfg.target_col].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df[cfg.target_col].astype(int).values

    # ─── MLflow Experiment Tracking ───────────────────────────────────────────
    mlflow.set_experiment("scam-detection-baseline")

    with mlflow.start_run(run_name="logistic-regression-saga"):
        # Log hyperparameters
        mlflow.log_params({
            "model": "LogisticRegression",
            "solver": "saga",
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "hash_dim": cfg.hash_dim,
            "random_state": cfg.random_state,
            "label_horizon_h": 72,
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
        })

        logger.info("Training LogisticRegression baseline")
        pipe.fit(X_train, y_train)

        logger.info("Evaluating on validation/test")
        valid_score = pipe.predict_proba(X_valid)[:, 1]
        test_score = pipe.predict_proba(X_test)[:, 1]

        valid_metrics = evaluate_binary(y_valid, valid_score)
        test_metrics = evaluate_binary(y_test, test_score)
        thr = best_f1_threshold(y_valid, valid_score)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "valid_auc_roc": valid_metrics["auc_roc"],
            "valid_avg_precision": valid_metrics["avg_precision"],
            "test_auc_roc": test_metrics["auc_roc"],
            "test_avg_precision": test_metrics["avg_precision"],
            "best_f1_threshold": thr["best_threshold"],
            "best_f1": thr["best_f1"],
            "precision_at_best_f1": thr["precision_at_best"],
            "recall_at_best_f1": thr["recall_at_best"],
        })

        # Persist model artefact
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path)
        mlflow.sklearn.log_model(pipe, "baseline_lr_model")
        mlflow.log_artifact(str(model_path))

        report = {
            "model": "logistic_regression_saga",
            "columns_used": cols_used,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
            "threshold_selection": thr,
            "notes": "Numeric + one-hot categoricals + hashed keys (entity_key, fingerprint_key).",
        }
        save_json(report, report_path)
        mlflow.log_artifact(str(report_path))

        logger.info(
            "Done. valid_auc=%.4f valid_ap=%.4f test_auc=%.4f test_ap=%.4f best_f1=%.4f",
            valid_metrics["auc_roc"],
            valid_metrics["avg_precision"],
            test_metrics["auc_roc"],
            test_metrics["avg_precision"],
            thr["best_f1"],
        )
        logger.info("MLflow run logged. Run 'make mlflow-ui' to explore.")


if __name__ == "__main__":
    main()
