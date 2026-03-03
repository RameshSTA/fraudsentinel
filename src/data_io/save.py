from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    """
    Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame to Parquet.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination parquet file path.
    """
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)