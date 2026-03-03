from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ProjectPaths, InterimFiles
from src.audit.audit_report import AuditConfig, build_audit_report, save_json
from src.utils.logging import setup_logger


def main() -> None:
    """
    Audit the merged dataset produced by scripts.01_load_merge.

    Input:
        data/interim/01_merged.parquet

    Output:
        reports/audit_01_merged.json
    """
    logger = setup_logger()

    paths = ProjectPaths()
    interim = InterimFiles()

    in_path = paths.data_interim / interim.merged_parquet
    out_path = paths.reports / "audit_01_merged.json"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing merged dataset: {in_path}. Run scripts.01_load_merge first.")

    logger.info("Loading merged parquet from %s", in_path)
    df = pd.read_parquet(in_path)

    logger.info("Building audit report")
    cfg = AuditConfig()
    report = build_audit_report(df, cfg)

    logger.info("Saving audit report to %s", out_path)
    save_json(report, out_path)

    # Print a short console summary
    rows = report["shape"]["rows"]
    cols = report["shape"]["columns"]
    fraud_rate = (report.get("label") or {}).get("isFraud_rate_pct")
    identity_pct = (report.get("identity_coverage") or {}).get("rows_with_any_identity_pct")

    logger.info("Audit summary: rows=%d cols=%d", rows, cols)
    if fraud_rate is not None:
        logger.info("Audit summary: isFraud_rate_pct=%.4f", fraud_rate)
    if identity_pct is not None:
        logger.info("Audit summary: identity_coverage_pct=%.2f", identity_pct)


if __name__ == "__main__":
    main()