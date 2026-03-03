from __future__ import annotations

from pathlib import Path

from src.config import ProjectPaths, RawFiles, InterimFiles
from src.data_io.load import load_csv, merge_on_transaction_id
from src.data_io.save import save_parquet
from src.utils.logging import setup_logger


def run_load_merge() -> Path:
    """
    Load raw IEEE-CIS CSVs, merge them on TransactionID, and save a merged parquet.

    Returns
    -------
    Path
        Path to the saved merged parquet file.
    """
    logger = setup_logger()

    paths = ProjectPaths()
    raw = RawFiles()
    interim = InterimFiles()

    txn_path = paths.data_raw / raw.train_transaction_csv
    id_path = paths.data_raw / raw.train_identity_csv
    out_path = paths.data_interim / interim.merged_parquet

    logger.info("Loading transactions from %s", txn_path)
    tx = load_csv(txn_path)

    logger.info("Loading identities from %s", id_path)
    idn = load_csv(id_path)

    logger.info("Merging dataframes (left join) on TransactionID")
    merged = merge_on_transaction_id(tx, idn)

    logger.info("Saving merged dataset to %s", out_path)
    save_parquet(merged, out_path)

    logger.info("Merged dataset saved. rows=%d cols=%d", len(merged), merged.shape[1])
    return out_path


def main() -> None:
    """
    CLI entrypoint for the load+merge pipeline step.
    """
    run_load_merge()


if __name__ == "__main__":
    main()