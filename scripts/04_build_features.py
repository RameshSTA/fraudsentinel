from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.build_features import FeatureConfig, build_features
from src.data_io.save import save_parquet
from src.utils.logging import setup_logger
from src.config import ProjectPaths


def _save_json(obj: dict, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Step 4: Feature engineering.

    Input:
        data/interim/02_cleaned.parquet

    Outputs:
        data/interim/03_features.parquet
        reports/features_03_features.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    in_path = paths.data_interim / "02_cleaned.parquet"
    out_path = paths.data_interim / "03_features.parquet"
    report_path = paths.reports / "features_03_features.json"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run Step 3 first.")

    logger.info("Loading cleaned dataset from %s", in_path)
    df = pd.read_parquet(in_path)

    logger.info("Building features (entity + propagation)")
    cfg = FeatureConfig()
    df_feat, report = build_features(df, cfg)

    logger.info("Saving features parquet to %s", out_path)
    save_parquet(df_feat, out_path)

    logger.info("Saving feature report to %s", report_path)
    _save_json(report, report_path)

    logger.info("Feature engineering complete. rows=%d cols=%d", report["rows"], report["cols"])
    logger.info("Engineered columns: %s", ", ".join(report["engineered_columns_present"]))


if __name__ == "__main__":
    main()