from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.inference.torch_infer import load_torch_artifact, predict_proba_torch
from src.utils.logging import setup_logger


def load_temperature(path: str | Path = "models/torch_temperature.json") -> float | None:
    """
    Load temperature scaling factor T if present; otherwise return None.
    """
    p = Path(path)
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    t = obj.get("temperature")
    if t is None:
        return None

    t = float(t)
    if t <= 0:
        return None
    return t


def assign_risk_band(score: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    """
    Assign risk band based on score thresholds.
    """
    s = score.astype(float)
    band = np.where(
        s >= cutoffs["critical"],
        "critical",
        np.where(
            s >= cutoffs["high"],
            "high",
            np.where(s >= cutoffs["medium"], "medium", "low"),
        ),
    )
    return pd.Series(band, index=score.index, dtype="string")


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    """
    Batch inference CLI.

    Example:
      python -m scripts.11_infer_batch \
        --input data/new/new_transactions.parquet \
        --output data/outputs/scored_new_transactions.parquet
    """
    parser = argparse.ArgumentParser(description="Batch inference: scam risk scoring")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--model", default="models/torch_mlp.pt", help="Path to torch model artifact (.pt)")
    parser.add_argument(
        "--temperature",
        default="models/torch_temperature.json",
        help="Path to temperature json (optional)",
    )
    parser.add_argument(
        "--policy",
        choices=["1_3_8", "0.5_2_5"],
        default="1_3_8",
        help="Risk band policy: 1/3/8 (recommended) or 0.5/2/5 (older)",
    )
    args = parser.parse_args()

    logger = setup_logger()

    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    logger.info("Loading input data: %s", input_path)
    df = pd.read_parquet(input_path)

    logger.info("Loading model artifact: %s", model_path)
    art = load_torch_artifact(model_path)

    # We will validate required columns using what the model artifact expects,
    # rather than a hardcoded list from config.
    required_cols = ["entity_key", "fingerprint_key", *art.numeric_cols]
    require_columns(df, required_cols)

    T = load_temperature(args.temperature)
    if T is None:
        logger.info("No temperature scaling found (or invalid). Using uncalibrated probabilities.")
    else:
        logger.info("Using temperature scaling: T=%.6f", T)

    logger.info("Scoring rows=%d", len(df))
    df["score"] = predict_proba_torch(df, art, temperature=T).astype(float)

    # Risk band cutoffs based on score percentiles of the scored batch
    if args.policy == "1_3_8":
        # critical top 1%, high top 3%, medium top 8%
        q_critical, q_high, q_medium = 0.99, 0.97, 0.92
    else:
        # critical top 0.5%, high top 2%, medium top 5%
        q_critical, q_high, q_medium = 0.995, 0.98, 0.95

    cutoffs = {
        "critical": float(df["score"].quantile(q_critical)),
        "high": float(df["score"].quantile(q_high)),
        "medium": float(df["score"].quantile(q_medium)),
    }

    df["risk_band"] = assign_risk_band(df["score"], cutoffs)

    logger.info("Saving scored output: %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(
        "Inference complete | rows=%d | critical=%d | high=%d | medium=%d | low=%d",
        len(df),
        int((df["risk_band"] == "critical").sum()),
        int((df["risk_band"] == "high").sum()),
        int((df["risk_band"] == "medium").sum()),
        int((df["risk_band"] == "low").sum()),
    )


if __name__ == "__main__":
    main()