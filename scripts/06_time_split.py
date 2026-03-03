"""

========================
Stage 6 — Temporal train / validation / test split.

Assigns every row to exactly one of three non-overlapping time windows based
on the ``TransactionDT`` column, using quantile-derived cutoffs:

    train  — first 70% of timeline
    valid  — next 15% (used for early stopping and threshold calibration)
    test   — final 15% (held out; never seen during training or tuning)

Using time-based splits rather than random splits is essential for payment
fraud: random splits allow future entity behaviour to leak into training,
producing over-optimistic metrics that do not reflect real deployment
conditions where the model always predicts on future traffic.

Input
-----
data/interim/04_labeled.parquet

Outputs
-------
data/processed/dataset_v1.parquet   — dataset with 'split' column appended
reports/split_v1.json               — cutoff values, row counts per split, fraud rates
"""

from __future__ import annotations

import pandas as pd

from src.config import ProjectPaths
from src.split.time_split import SplitConfig, assign_split, build_split_report, compute_cutoffs
from src.data_io.save import save_parquet
from src.utils.json_utils import save_json
from src.utils.logging import setup_logger


def main() -> None:
    """
    Step 6: Time-based train/valid/test split.

    Input:
        data/interim/04_labeled.parquet

    Outputs:
        data/processed/dataset_v1.parquet
        reports/split_v1.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    in_path = paths.data_interim / "04_labeled.parquet"
    out_path = paths.data_processed / "dataset_v1.parquet"
    report_path = paths.reports / "split_v1.json"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run Step 5 first.")

    logger.info("Loading labeled dataset: %s", in_path)
    df = pd.read_parquet(in_path)

    cfg = SplitConfig(train_quantile=0.70, valid_quantile=0.85, time_col="TransactionDT")

    logger.info("Computing time cutoffs using quantiles: train=%.2f valid=%.2f", cfg.train_quantile, cfg.valid_quantile)
    train_end, valid_end = compute_cutoffs(df[cfg.time_col], cfg)

    logger.info("Assigning split labels using train_end=%d valid_end=%d", train_end, valid_end)
    df_out = df.copy()
    df_out["split"] = assign_split(df_out[cfg.time_col], train_end, valid_end)

    # Build report
    report = {
        "time_col": cfg.time_col,
        "train_quantile": cfg.train_quantile,
        "valid_quantile": cfg.valid_quantile,
        "train_end": int(train_end),
        "valid_end": int(valid_end),
        "dt_min": int(pd.to_numeric(df_out[cfg.time_col], errors="coerce").min()),
        "dt_max": int(pd.to_numeric(df_out[cfg.time_col], errors="coerce").max()),
    }
    report.update(build_split_report(df_out, split_col="split", label_col="y_ew_72h"))

    logger.info("Saving processed dataset: %s", out_path)
    save_parquet(df_out, out_path)

    logger.info("Saving split report: %s", report_path)
    save_json(report, report_path)

    logger.info("Split complete. Split counts: %s", report["split_counts"])


if __name__ == "__main__":
    main()