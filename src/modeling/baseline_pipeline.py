from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class BaselineConfig:
    """
    Baseline model configuration.

    Parameters
    ----------
    target_col:
        Name of the label column.
    split_col:
        Column indicating train/valid/test split.
    hash_dim:
        Dimensionality used for hashing high-cardinality keys.
    random_state:
        Random seed for reproducibility.
    """
    target_col: str = "y_ew_72h"
    split_col: str = "split"
    hash_dim: int = 2**18
    random_state: int = 42


class KeyHasher(BaseEstimator, TransformerMixin):
    """
    Hash high-cardinality string columns into a fixed-size sparse feature space.

    Why this exists
    ---------------
    Inside sklearn ColumnTransformer, downstream transformers may receive either:
    - pandas DataFrame (with column names), or
    - NumPy array (no column names)

    This transformer handles both safely.

    Output
    ------
    scipy sparse matrix (FeatureHasher output), suitable for linear models.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self._hasher = FeatureHasher(n_features=n_features, input_type="dict")
        self._col_names: list[str] | None = None

    def fit(self, X, y=None):
        # If pandas DataFrame, keep column names. If numpy array, create synthetic names.
        if hasattr(X, "columns"):
            self._col_names = [str(c) for c in X.columns]
        else:
            self._col_names = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if self._col_names is None:
            raise RuntimeError("KeyHasher must be fit() before transform().")

        rows: list[dict[str, int]] = []

        if hasattr(X, "iterrows"):
            # pandas DataFrame
            for _, r in X.iterrows():
                d: dict[str, int] = {}
                for c in self._col_names:
                    val = r[c] if c in r.index else ""
                    v = "" if pd.isna(val) else str(val)
                    d[f"{c}={v}"] = 1
                rows.append(d)
        else:
            # numpy array
            for r in X:
                d = {}
                for c, val in zip(self._col_names, r):
                    v = "" if pd.isna(val) else str(val)
                    d[f"{c}={v}"] = 1
                rows.append(d)

        return self._hasher.transform(rows)


def build_baseline_pipeline(
    df: pd.DataFrame,
    cfg: BaselineConfig,
) -> tuple[Pipeline, dict[str, list[str]]]:
    """
    Build preprocessing + baseline model pipeline.

    Strategy
    --------
    - Numeric features: median impute + scale
    - Categorical (small): most-frequent impute + one-hot
    - High-cardinality keys: constant impute + hashing

    Returns
    -------
    (pipeline, columns_used)
        sklearn Pipeline and a dict of chosen columns.
    """
    numeric_cols = [
        "TransactionAmt",
        "cnt_1h", "sum_amt_1h",
        "cnt_24h", "avg_amt_24h",
        "mean_amt_7d_hist", "std_amt_7d_hist", "z_amt_vs_7d",
        "fp_cnt_24h", "fp_cnt_72h", "fp_growth_ratio_24h_over_72h",
    ]
    categorical_cols = ["ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType"]
    hashed_key_cols = ["entity_key", "fingerprint_key"]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    hashed_key_cols = [c for c in hashed_key_cols if c in df.columns]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    key_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("hasher", KeyHasher(n_features=cfg.hash_dim)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
            ("keyhash", key_pipe, hashed_key_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=cfg.random_state,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    columns_used = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "hashed_keys": hashed_key_cols,
    }
    return pipe, columns_used


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """
    Evaluate imbalanced binary classification with AUC-ROC and Average Precision.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """
    Pick threshold that maximizes F1 on a validation set.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    best_idx = int(np.nanargmax(f1))
    thr = float(thresholds[max(best_idx - 1, 0)]) if thresholds.size else 0.5

    return {
        "best_threshold": thr,
        "best_f1": float(f1[best_idx]),
        "precision_at_best": float(precision[best_idx]),
        "recall_at_best": float(recall[best_idx]),
    }