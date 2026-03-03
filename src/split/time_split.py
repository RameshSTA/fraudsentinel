from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """
    Time-based split configuration.

    Parameters
    ----------
    train_quantile:
        Quantile cutoff for train end based on TransactionDT.
    valid_quantile:
        Quantile cutoff for valid end based on TransactionDT.
    time_col:
        Name of the time column (seconds).
    """
    train_quantile: float = 0.70
    valid_quantile: float = 0.85
    time_col: str = "TransactionDT"


def compute_cutoffs(dt: pd.Series, cfg: SplitConfig) -> tuple[int, int]:
    """
    Compute train/valid cutoffs as quantiles of TransactionDT.

    Parameters
    ----------
    dt:
        Series of timestamps in seconds.
    cfg:
        SplitConfig.

    Returns
    -------
    (train_end, valid_end)
        Integer time cutoffs.
    """
    dt_num = pd.to_numeric(dt, errors="coerce").dropna().astype("int64")
    if dt_num.empty:
        raise ValueError("TransactionDT has no valid numeric values.")

    train_end = int(dt_num.quantile(cfg.train_quantile, interpolation="higher"))
    valid_end = int(dt_num.quantile(cfg.valid_quantile, interpolation="higher"))

    if valid_end < train_end:
        raise ValueError("Computed valid_end is earlier than train_end. Check quantiles.")

    return train_end, valid_end


def assign_split(dt: pd.Series, train_end: int, valid_end: int) -> pd.Series:
    """
    Assign 'train'/'valid'/'test' based on TransactionDT cutoffs.

    Parameters
    ----------
    dt:
        TransactionDT series.
    train_end:
        Max TransactionDT for train.
    valid_end:
        Max TransactionDT for valid.

    Returns
    -------
    pd.Series
        Split labels aligned to input.
    """
    dt_num = pd.to_numeric(dt, errors="coerce").astype("int64")

    split = np.where(
        dt_num <= train_end,
        "train",
        np.where(dt_num <= valid_end, "valid", "test"),
    )
    return pd.Series(split, index=dt.index, dtype="string")


def build_split_report(df: pd.DataFrame, *, split_col: str, label_col: str) -> dict[str, Any]:
    """
    Build a split distribution report.

    Parameters
    ----------
    df:
        Dataframe containing split and label columns.
    split_col:
        Column name for split labels.
    label_col:
        Column name for target label.

    Returns
    -------
    dict[str, Any]
        JSON-serializable report.
    """
    report: dict[str, Any] = {}

    report["rows"] = int(len(df))
    report["split_counts"] = df[split_col].value_counts(dropna=False).to_dict()

    if label_col in df.columns:
        tmp = df[[split_col, label_col]].copy()
        tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce").fillna(0).astype("int64")

        grp = tmp.groupby(split_col, dropna=False)[label_col].agg(["count", "sum", "mean"])
        report["label_by_split"] = {
            str(idx): {
                "count": int(row["count"]),
                "positive": int(row["sum"]),
                "positive_rate_pct": float(100.0 * row["mean"]),
            }
            for idx, row in grp.iterrows()
        }

    return report