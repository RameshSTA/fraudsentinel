from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    """
    Labeling configuration.

    Parameters
    ----------
    horizon_seconds:
        Prediction horizon. For early warning, we predict if fraud occurs within
        the next horizon_seconds for the same entity_key.
    """
    horizon_seconds: int = 72 * 3600


def make_early_warning_label(
    df: pd.DataFrame,
    cfg: LabelConfig,
    *,
    entity_col: str = "entity_key",
    time_col: str = "TransactionDT",
    fraud_col: str = "isFraud",
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Create a time-safe early-warning label for each event.

    Definition
    ----------
    y_ew = 1 if there exists a fraud event for the same entity occurring at time t_f
    such that: 0 < (t_f - t) <= horizon_seconds.

    This implementation is efficient:
    - Sort within entity by time
    - Track the next fraud time per row
    - Compute whether the next fraud occurs within the horizon

    Parameters
    ----------
    df:
        Input dataframe (must include entity_col, time_col, fraud_col).
    cfg:
        LabelConfig containing the prediction horizon.
    entity_col:
        Column name representing the entity grouping key.
    time_col:
        Column name containing event timestamps in seconds.
    fraud_col:
        Column indicating whether the current event is fraud (0/1).

    Returns
    -------
    (pd.Series, dict[str, Any])
        y_ew label series (int8) aligned to df.index, and summary metrics.
    """
    required = {entity_col, time_col, fraud_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for labeling: {sorted(missing)}")

    work = df[[entity_col, time_col, fraud_col]].copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce").astype("int64")
    work[fraud_col] = pd.to_numeric(work[fraud_col], errors="coerce").fillna(0).astype("int8")

    # Keep original index alignment
    work["_orig_idx"] = work.index

    # Sort by entity, then time
    work = work.sort_values([entity_col, time_col], kind="mergesort").reset_index(drop=True)

    # Fraud times for each row (NaN where not fraud)
    fraud_time = work[time_col].where(work[fraud_col] == 1, np.nan)

    # Next fraud time including current row: reverse cumulative min within each group
    next_fraud_incl = (
        fraud_time.groupby(work[entity_col], sort=False)
        .apply(lambda s: s[::-1].cummin()[::-1])
        .reset_index(level=0, drop=True)
    )

    # Shift so that a fraud event does not label itself; we want the *future* fraud
    next_fraud_future = next_fraud_incl.groupby(work[entity_col], sort=False).shift(-1)

    delta = next_fraud_future - work[time_col]
    y = ((delta > 0) & (delta <= cfg.horizon_seconds)).astype("int8")

    # Restore original ordering
    labeled = pd.Series(y.values, index=work["_orig_idx"].values).reindex(df.index).astype("int8")

    metrics: dict[str, Any] = {
        "horizon_seconds": int(cfg.horizon_seconds),
        "label_name": "y_ew_72h",
        "positive_rate_pct": float(100.0 * labeled.mean()),
        "positive_count": int(labeled.sum()),
        "total_rows": int(len(labeled)),
    }

    return labeled, metrics