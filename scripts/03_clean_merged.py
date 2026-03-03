"""
==========================
Stage 3 — Defensive data cleaning.

Applies a minimal, well-documented set of cleaning rules to the merged
transaction+identity dataset produced by Stage 1. The goal is to resolve
genuine data-quality issues (duplicate primary keys, wrong dtypes) without
silently discarding information that may be analytically relevant.

Input
-----
data/interim/01_merged.parquet

Outputs
-------
data/interim/02_cleaned.parquet   — cleaned dataset
reports/cleaning_02_cleaned.json  — row counts, null rates, label statistics
"""

from __future__ import annotations

import pandas as pd

from src.config import InterimFiles, ProjectPaths
from src.cleaning.clean import CleaningConfig, clean_merged_dataset
from src.data_io.save import save_parquet
from src.utils.json_utils import save_json
from src.utils.logging import setup_logger


def main() -> None:
    """
    Step 3: Clean the merged dataset.

    Input:
        data/interim/01_merged.parquet

    Outputs:
        data/interim/02_cleaned.parquet
        reports/cleaning_02_cleaned.json
    """
    logger = setup_logger()

    paths = ProjectPaths()
    interim = InterimFiles()

    input_path = paths.data_interim / interim.merged_parquet
    output_path = paths.data_interim / "02_cleaned.parquet"
    report_path = paths.reports / "cleaning_02_cleaned.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}. Run Step 1 first.")

    logger.info("Reading merged dataset: %s", input_path)
    df = pd.read_parquet(input_path)

    logger.info("Running cleaning step with minimal, safe rules")
    cfg = CleaningConfig(enforce_transaction_id_unique=True, cast_types=True)
    cleaned, metrics = clean_merged_dataset(df, cfg)

    logger.info("Writing cleaned dataset: %s", output_path)
    save_parquet(cleaned, output_path)

    logger.info("Writing cleaning report: %s", report_path)
    save_json(metrics, report_path)

    logger.info(
        "Cleaning complete: rows_before=%d rows_after=%d duplicate_txid_rows=%d label_rate_pct=%s identity_coverage_pct=%s",
        metrics["rows_before"],
        metrics["rows_after"],
        metrics.get("duplicate_transaction_id_rows", 0),
        f"{metrics['label_rate_pct']:.4f}" if metrics["label_rate_pct"] is not None else "None",
        f"{metrics['identity_coverage_pct']:.2f}" if metrics.get("identity_coverage_pct") is not None else "None",
    )


if __name__ == "__main__":
    main()