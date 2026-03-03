from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.config import ProjectPaths
from src.inference.torch_infer import load_torch_artifact, predict_proba_torch
from src.utils.logging import setup_logger


def _load_temperature(path: str | Path = "models/torch_temperature.json") -> float | None:
    """
    Load temperature scaling factor T if it exists.
    Returns None if file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    T = obj.get("temperature", None)
    if T is None:
        return None

    T = float(T)
    if T <= 0:
        return None
    return T


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def assign_risk_band(score: pd.Series, cutoffs: dict[str, float]) -> pd.Series:
    """
    Assign risk band based on score thresholds.

    Bands:
      - critical: score >= critical_threshold
      - high:     score >= high_threshold
      - medium:   score >= medium_threshold
      - low:      otherwise
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


def band_summary(df: pd.DataFrame, label_col: str = "y_ew_72h") -> dict:
    """
    Build per-band and cumulative performance summary.

    Notes:
      - Per-band precision = mean(label) within band
      - Cumulative metrics are computed by sorting bands in descending risk
    """
    out: dict = {}

    # Per band (keep stable ordering later via explicit categories)
    g = df.groupby("risk_band")[label_col].agg(["count", "sum", "mean"])
    per_band = {
        str(k): {
            "count": int(v["count"]),
            "positives": int(v["sum"]),
            "precision": float(v["mean"]),
        }
        for k, v in g.iterrows()
    }
    out["per_band"] = per_band

    # Cumulative (top risk first)
    order = ["critical", "high", "medium", "low"]
    df2 = df.copy()
    df2["risk_band"] = pd.Categorical(df2["risk_band"], categories=order, ordered=True)
    df2 = df2.sort_values(["risk_band", "score"], ascending=[True, False])

    total_pos = int(df2[label_col].sum())
    total = int(len(df2))

    cum = []
    running_pos = 0
    running_total = 0

    for band in order:
        sub = df2[df2["risk_band"] == band]
        running_total += int(len(sub))
        running_pos += int(sub[label_col].sum())

        cum.append(
            {
                "up_to_band": band,
                "rows": running_total,
                "rows_pct": float(100.0 * running_total / total) if total > 0 else None,
                "captured_pos": running_pos,
                "recall": float(running_pos / total_pos) if total_pos > 0 else None,
                "precision": float(running_pos / running_total) if running_total > 0 else None,
            }
        )

    out["cumulative"] = cum
    out["total_rows"] = total
    out["total_positives"] = total_pos
    out["base_rate"] = float(total_pos / total) if total > 0 else None
    return out


def main() -> None:
    """
    Step 9: Create capacity-based risk bands and export scored test set.

    Inputs:
      - data/processed/dataset_v1.parquet
      - models/torch_mlp.pt
      - (optional) models/torch_temperature.json

    Outputs:
      - data/outputs/test_scored.parquet
      - reports/risk_band_summary.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    data_path = paths.data_processed / "dataset_v1.parquet"
    model_path = Path("models/torch_mlp.pt")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing torch model: {model_path}")

    logger.info("Loading dataset: %s", data_path)
    df = pd.read_parquet(data_path)

    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test rows found. Check your split step.")

    logger.info("Loading torch model: %s", model_path)
    art = load_torch_artifact(model_path)

    # Optional temperature scaling
    T = _load_temperature("models/torch_temperature.json")
    if T is None:
        logger.info("No temperature scaling file found. Using uncalibrated torch probabilities.")
    else:
        logger.info("Using temperature scaling: T=%.6f", T)

    logger.info("Scoring test set with torch model")
    test_df["score"] = predict_proba_torch(test_df, art, temperature=T).astype(float)

    # More realistic ops tiers (less noisy than 0.5/2/5)
    # Interpretations:
    #  - critical: must-review queue (e.g., high urgency)
    #  - high:     daily review queue
    #  - medium:   automated + sampled review / rules
    p_critical = 0.01  # top 1%
    p_high = 0.03      # top 3%
    p_medium = 0.08    # top 8%

    # Convert top-pct into score thresholds
    cutoffs = {
        "critical": float(test_df["score"].quantile(1.0 - p_critical)),
        "high": float(test_df["score"].quantile(1.0 - p_high)),
        "medium": float(test_df["score"].quantile(1.0 - p_medium)),
    }

    logger.info("Risk band cutoffs: %s", cutoffs)

    test_df["risk_band"] = assign_risk_band(test_df["score"], cutoffs)

    summary = {
        "policy": {
            "type": "capacity_based",
            "bands": {
                "critical_top_pct": p_critical * 100.0,
                "high_top_pct": p_high * 100.0,
                "medium_top_pct": p_medium * 100.0,
            },
            "temperature_used": T,
            "score_cutoffs": cutoffs,
        },
        "summary": band_summary(test_df, label_col="y_ew_72h"),
    }

    out_scored = Path("data/outputs/test_scored.parquet")
    out_report = paths.reports / "risk_band_summary.json"

    logger.info("Saving scored test set: %s", out_scored)
    out_scored.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(out_scored, index=False)

    logger.info("Saving risk band report: %s", out_report)
    save_json(summary, out_report)

    logger.info("Done. Wrote %s and %s", out_scored, out_report)


if __name__ == "__main__":
    main()