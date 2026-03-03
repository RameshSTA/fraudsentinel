"""
==================================
Stage 5 — Early-warning label construction.

Creates the binary target ``y_ew_72h``: a transaction is labelled positive
if a confirmed fraud event is observed for the *same entity* within the next
72 hours. This reframes the modelling task from reactive detection to
proactive early-warning, enabling interventions before losses occur.

Label definition
----------------
    y_ew_72h = 1  iff  ∃ fraud event for same entity within (0, 72h] after T
    y_ew_72h = 0  otherwise

Leakage safeguards
------------------
- The current transaction's own ``isFraud`` flag is excluded (shift -1 applied
  inside :func:`make_early_warning_label`).
- Label is computed from entity-sorted transaction order, so no future data
  pollutes rows earlier in time.

Input
-----
data/interim/03_features.parquet

Outputs
-------
data/interim/04_labeled.parquet      — dataset with y_ew_72h column appended
reports/labels_04_labeled.json       — positive rate, counts, horizon seconds
"""

from __future__ import annotations

import pandas as pd

from src.config import ProjectPaths
from src.labels.early_warning import LabelConfig, make_early_warning_label
from src.data_io.save import save_parquet
from src.utils.json_utils import save_json
from src.utils.logging import setup_logger


def main() -> None:
    """
    Step 5: Create early-warning label y_ew_72h.

    Input:
        data/interim/03_features.parquet

    Outputs:
        data/interim/04_labeled.parquet
        reports/labels_04_labeled.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    in_path = paths.data_interim / "03_features.parquet"
    out_path = paths.data_interim / "04_labeled.parquet"
    report_path = paths.reports / "labels_04_labeled.json"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run Step 4 first.")

    logger.info("Loading features dataset: %s", in_path)
    df = pd.read_parquet(in_path)

    logger.info("Creating early-warning label (72h horizon)")
    cfg = LabelConfig(horizon_seconds=72 * 3600)
    y, metrics = make_early_warning_label(df, cfg)

    df_out = df.copy()
    df_out["y_ew_72h"] = y

    logger.info("Saving labeled dataset: %s", out_path)
    save_parquet(df_out, out_path)

    logger.info("Saving label report: %s", report_path)
    save_json(metrics, report_path)

    logger.info(
        "Labeling complete: positive_rate_pct=%.4f positive_count=%d total=%d",
        metrics["positive_rate_pct"],
        metrics["positive_count"],
        metrics["total_rows"],
    )


if __name__ == "__main__":
    main()