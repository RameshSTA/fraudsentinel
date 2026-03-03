from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_csv(path: Path, *, low_memory: bool = False) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.
    low_memory:
        Whether pandas should internally process the file in chunks to reduce memory usage.
        Keeping this False generally gives more stable dtype inference.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path, low_memory=low_memory)


def merge_on_transaction_id(
    transactions: pd.DataFrame,
    identities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge transaction and identity data using a left join on TransactionID.

    Parameters
    ----------
    transactions:
        Transaction-level dataframe (must contain TransactionID).
    identities:
        Identity-level dataframe (must contain TransactionID).

    Returns
    -------
    pd.DataFrame
        Merged dataframe.

    Raises
    ------
    ValueError
        If TransactionID is missing in either dataframe.
    """
    if "TransactionID" not in transactions.columns:
        raise ValueError("transactions is missing required column: TransactionID")
    if "TransactionID" not in identities.columns:
        raise ValueError("identities is missing required column: TransactionID")

    merged = transactions.merge(identities, on="TransactionID", how="left")
    return merged