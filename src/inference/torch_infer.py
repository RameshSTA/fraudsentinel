"""
============================
Production inference utilities for the Torch MLP fraud model.

This module owns the full prediction path from raw feature DataFrame to
calibrated fraud probability:

  1. Load the serialised ``.pt`` artefact (weights + config + scaler stats).
  2. Reconstruct the :class:`TorchMLP` in inference mode.
  3. Apply the same numeric standardisation and deterministic key hashing used
     during training.
  4. Run the forward pass in ``torch.no_grad()`` context for efficiency.
  5. Optionally divide logits by the temperature scalar ``T`` before sigmoid
     to produce calibrated probabilities.

The :class:`TorchArtifact` dataclass bundles everything the inference path
needs into a single immutable object, making it safe to load once at server
startup and share across concurrent requests.

Device priority
---------------
MPS (Apple Silicon) → CUDA (NVIDIA GPU) → CPU
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.modeling.torch_mlp import TorchMLP, stable_hash_to_bucket


@dataclass(frozen=True)
class TorchArtifact:
    model: torch.nn.Module
    numeric_cols: list[str]
    scaler_stats: dict[str, Any]
    entity_buckets: int
    fingerprint_buckets: int
    device: torch.device


def _transform_numeric(df: pd.DataFrame, numeric_cols: list[str], stats: dict[str, Any]) -> np.ndarray:
    """
    Standardize numeric columns with pre-fit stats; impute missing to 0 after standardization.
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


def _transform_key_to_buckets(series: pd.Series, buckets: int) -> np.ndarray:
    s = series.astype("string").fillna("")
    return np.fromiter((stable_hash_to_bucket(v, buckets) for v in s.tolist()), dtype=np.int64)


def load_torch_artifact(model_path: str | Path) -> TorchArtifact:
    """
    Load a trained TorchMLP artifact saved by scripts/08_train_torch_mlp.py.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing torch model file: {model_path}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    payload = torch.load(model_path, map_location="cpu")

    cfg = payload["config"]
    numeric_cols = payload["numeric_cols"]
    scaler_stats = payload["scaler_stats"]
    entity_buckets = int(payload["entity_buckets"])
    fingerprint_buckets = int(payload["fingerprint_buckets"])

    model = TorchMLP(
        num_dim=len(numeric_cols),
        entity_buckets=entity_buckets,
        fingerprint_buckets=fingerprint_buckets,
        entity_emb_dim=int(cfg["entity_emb_dim"]),
        fingerprint_emb_dim=int(cfg["fingerprint_emb_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    )

    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    return TorchArtifact(
        model=model,
        numeric_cols=numeric_cols,
        scaler_stats=scaler_stats,
        entity_buckets=entity_buckets,
        fingerprint_buckets=fingerprint_buckets,
        device=device,
    )


@torch.no_grad()
def predict_proba_torch(df: pd.DataFrame, art: TorchArtifact, temperature: float | None = None) -> np.ndarray:
    """
    Predict probabilities for a dataframe using the Torch artifact.

    Requires columns:
      - numeric_cols from artifact
      - entity_key
      - fingerprint_key
    """
    required = set(art.numeric_cols) | {"entity_key", "fingerprint_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for torch inference: {sorted(missing)}")

    X_num = _transform_numeric(df, art.numeric_cols, art.scaler_stats)
    x_ent = _transform_key_to_buckets(df["entity_key"], art.entity_buckets)
    x_fp = _transform_key_to_buckets(df["fingerprint_key"], art.fingerprint_buckets)

    x_num_t = torch.from_numpy(X_num).to(art.device)
    x_ent_t = torch.from_numpy(x_ent).long().to(art.device)
    x_fp_t = torch.from_numpy(x_fp).long().to(art.device)
    
    logits = art.model(x_num_t, x_ent_t, x_fp_t)
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs

def load_temperature(path: str | Path = "models/torch_temperature.json") -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    import json
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj.get("temperature", 1.0))