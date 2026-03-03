"""

=========================
Torch MLP architecture, dataset class, and training utilities.

Architecture overview
---------------------
The model combines three input streams:

  Numeric stream (11 features)
    Raw transaction amounts, velocity counts, behavioural z-scores, and
    fingerprint growth ratios — standardised with train-split statistics.

  Entity embedding stream
    ``entity_key`` (card1||card2||card3||card5||addr1) is hashed by MD5 to
    one of 1,048,576 buckets, then mapped through a 32-dimensional embedding
    table.  This lets the model learn a dense representation of each card/
    address composite's historical fraud propensity without an explicit
    vocabulary or high-dimensional one-hot encoding.

  Fingerprint embedding stream
    ``fingerprint_key`` (DeviceInfo||P_emaildomain||ProductCD) is hashed to
    one of 262,144 buckets and embedded into 16 dimensions.  Campaign-level
    propagation signals are captured here.

All three streams are concatenated into a 59-dimensional vector, which passes
through two fully-connected ReLU layers (256 → 128) with dropout before
producing a single raw logit.  Temperature scaling is applied post-hoc
(separate module) to calibrate the final probabilities.

Key design decisions
--------------------
- **MD5 hashing to buckets** ensures the same key always maps to the same
  embedding row — deterministic and dependency-free.
- **pos_weight in BCEWithLogitsLoss** corrects for the 34:1 class imbalance
  without requiring oversampling or undersampling.
- **Early stopping on validation AUC-PR** (not loss) prevents overfitting on
  the majority class while directly optimising the metric that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TorchConfig:
    """
    PyTorch training configuration.
    """
    target_col: str = "y_ew_72h"
    split_col: str = "split"

    # Hash buckets for embeddings (fixed-size vocab)
    entity_buckets: int = 2**20        # 1,048,576
    fingerprint_buckets: int = 2**18   # 262,144

    # Embedding dims
    entity_emb_dim: int = 32
    fingerprint_emb_dim: int = 16

    # MLP
    hidden_dim: int = 256
    dropout: float = 0.20

    # Training
    batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 3
    num_workers: int = 0  # Mac default


def stable_hash_to_bucket(value: str, buckets: int) -> int:
    """
    Deterministic hash -> [0, buckets).

    Uses md5 for stability across processes/runs/platforms.
    """
    if value is None:
        value = ""
    s = value.encode("utf-8", errors="ignore")
    h = hashlib.md5(s).hexdigest()
    return int(h, 16) % buckets


def build_feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    """
    Define which columns are used as numeric and categorical keys.

    We intentionally exclude raw high-dimensional columns. Your features table
    is already compact (21 cols), but we still keep a clear schema.
    """
    numeric_cols = [
        "TransactionAmt",
        "cnt_1h", "sum_amt_1h",
        "cnt_24h", "avg_amt_24h",
        "mean_amt_7d_hist", "std_amt_7d_hist", "z_amt_vs_7d",
        "fp_cnt_24h", "fp_cnt_72h", "fp_growth_ratio_24h_over_72h",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    key_cols = {
        "entity_key": "entity_key" if "entity_key" in df.columns else None,
        "fingerprint_key": "fingerprint_key" if "fingerprint_key" in df.columns else None,
    }

    if key_cols["entity_key"] is None or key_cols["fingerprint_key"] is None:
        raise ValueError("Missing entity_key and/or fingerprint_key. Run Step 4 feature engineering first.")

    return {"numeric_cols": numeric_cols, "key_cols": key_cols}


def fit_standardizer(train_df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """
    Fit simple standardization stats on train split only.
    """
    X = train_df[numeric_cols].copy()
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    means = X.mean(skipna=True).astype(float).to_dict()
    stds = X.std(skipna=True, ddof=0).replace(0.0, 1.0).astype(float).to_dict()

    return {"means": means, "stds": stds}


def transform_numeric(df: pd.DataFrame, numeric_cols: list[str], stats: dict[str, Any]) -> np.ndarray:
    """
    Standardize numeric columns using pre-fit stats.
    NaNs are imputed to 0 after standardization (i.e., mean-impute).
    """
    X = df[numeric_cols].copy()
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    means = stats["means"]
    stds = stats["stds"]

    for c in numeric_cols:
        X[c] = (X[c] - means[c]) / stds[c]

    X = X.fillna(0.0)
    return X.to_numpy(dtype=np.float32, copy=True)


def transform_keys(df: pd.DataFrame, col: str, buckets: int) -> np.ndarray:
    """
    Hash a string column into embedding bucket indices.
    """
    s = df[col].astype("string").fillna("")
    out = np.fromiter((stable_hash_to_bucket(v, buckets) for v in s.tolist()), dtype=np.int64)
    return out


class TabularFraudDataset(Dataset):
    """
    Torch Dataset for tabular data with two hashed embedding keys + numeric features.
    """

    def __init__(
        self,
        X_num: np.ndarray,
        x_entity: np.ndarray,
        x_fp: np.ndarray,
        y: np.ndarray | None,
    ) -> None:
        self.X_num = torch.from_numpy(X_num)                  # float32 [N, D]
        self.x_entity = torch.from_numpy(x_entity).long()     # int64 [N]
        self.x_fp = torch.from_numpy(x_fp).long()             # int64 [N]
        self.y = None if y is None else torch.from_numpy(y).float()  # float32 [N]

    def __len__(self) -> int:
        return self.X_num.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X_num[idx], self.x_entity[idx], self.x_fp[idx]
        return self.X_num[idx], self.x_entity[idx], self.x_fp[idx], self.y[idx]


class TorchMLP(nn.Module):
    """
    Numeric + (entity embedding) + (fingerprint embedding) -> MLP -> logit.
    """

    def __init__(
        self,
        num_dim: int,
        entity_buckets: int,
        fingerprint_buckets: int,
        entity_emb_dim: int,
        fingerprint_emb_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.entity_emb = nn.Embedding(entity_buckets, entity_emb_dim)
        self.fp_emb = nn.Embedding(fingerprint_buckets, fingerprint_emb_dim)

        in_dim = num_dim + entity_emb_dim + fingerprint_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize embeddings
        nn.init.normal_(self.entity_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fp_emb.weight, mean=0.0, std=0.01)

    def forward(self, x_num: torch.Tensor, x_entity: torch.Tensor, x_fp: torch.Tensor) -> torch.Tensor:
        e = self.entity_emb(x_entity)
        f = self.fp_emb(x_fp)
        x = torch.cat([x_num, e, f], dim=1)
        logit = self.net(x).squeeze(1)
        return logit


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)