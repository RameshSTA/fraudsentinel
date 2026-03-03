from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss

from src.config import ProjectPaths
from src.inference.torch_infer import load_torch_artifact
from src.utils.logging import setup_logger


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def predict_logits(df: pd.DataFrame, art) -> np.ndarray:
    """
    Return raw logits (before sigmoid) for temperature scaling.
    """
    required = set(art.numeric_cols) | {"entity_key", "fingerprint_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Reuse internal transforms from torch_infer artifact contents
    # We implement a minimal transform here to avoid circular dependencies.
    import numpy as np
    import pandas as pd
    import torch
    from src.modeling.torch_mlp import stable_hash_to_bucket

    def transform_numeric(df_: pd.DataFrame) -> np.ndarray:
        X = df_[art.numeric_cols].copy()
        for c in art.numeric_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        means = art.scaler_stats["means"]
        stds = art.scaler_stats["stds"]
        for c in art.numeric_cols:
            X[c] = (X[c] - means[c]) / stds[c]
        return X.fillna(0.0).to_numpy(dtype=np.float32, copy=True)

    def transform_key(series: pd.Series, buckets: int) -> np.ndarray:
        s = series.astype("string").fillna("")
        return np.fromiter((stable_hash_to_bucket(v, buckets) for v in s.tolist()), dtype=np.int64)

    X_num = transform_numeric(df)
    x_ent = transform_key(df["entity_key"], art.entity_buckets)
    x_fp = transform_key(df["fingerprint_key"], art.fingerprint_buckets)

    x_num_t = torch.from_numpy(X_num).to(art.device)
    x_ent_t = torch.from_numpy(x_ent).long().to(art.device)
    x_fp_t = torch.from_numpy(x_fp).long().to(art.device)

    logits = art.model(x_num_t, x_ent_t, x_fp_t).detach().cpu().numpy()
    return logits


def main() -> None:
    """
    Fit temperature scaling parameter T on the validation split.

    Inputs:
      - data/processed/dataset_v1.parquet
      - models/torch_mlp.pt

    Output:
      - models/torch_temperature.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    data_path = paths.data_processed / "dataset_v1.parquet"
    model_path = Path("models/torch_mlp.pt")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    df = pd.read_parquet(data_path)
    valid_df = df[df["split"] == "valid"].copy()
    if valid_df.empty:
        raise ValueError("No validation rows found.")

    y = valid_df["y_ew_72h"].astype(int).to_numpy()

    art = load_torch_artifact(model_path)
    logits_np = predict_logits(valid_df, art)

    # Learn temperature T by minimizing NLL on validation
    device = art.device
    logits = torch.tensor(logits_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    log_T = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        loss = bce(logits / T, y_t)
        loss.backward()
        return loss

    opt.step(closure)

    T = float(torch.exp(log_T).detach().cpu().item())

    # Report calibration improvement (Brier + logloss) on valid
    prob_uncal = 1 / (1 + np.exp(-logits_np))
    prob_cal = 1 / (1 + np.exp(-(logits_np / T)))

    report = {
        "temperature": T,
        "valid": {
            "brier_uncal": float(brier_score_loss(y, prob_uncal)),
            "brier_cal": float(brier_score_loss(y, prob_cal)),
            "logloss_uncal": float(log_loss(y, prob_uncal)),
            "logloss_cal": float(log_loss(y, prob_cal)),
        },
    }

    out_path = Path("models/torch_temperature.json")
    save_json(report, out_path)

    logger.info("Saved temperature scaling: %s", out_path)
    logger.info("T=%.6f | brier %.6f -> %.6f | logloss %.6f -> %.6f",
                T,
                report["valid"]["brier_uncal"],
                report["valid"]["brier_cal"],
                report["valid"]["logloss_uncal"],
                report["valid"]["logloss_cal"])


if __name__ == "__main__":
    main()