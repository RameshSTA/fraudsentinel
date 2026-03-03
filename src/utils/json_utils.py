"""
src/utils/json_utils.py
=======================
Shared JSON serialisation helpers used across the pipeline.

Background
----------
NumPy and Pandas introduce many types that are not natively serialisable by
Python's :mod:`json` module — ``np.int64``, ``np.float32``, ``np.bool_``,
``pd.NA``, ``pd.Timestamp``, etc.  Scattering ad-hoc conversion logic across
every script that writes a report creates duplication and inconsistency.  This
module provides one canonical implementation used everywhere.

Public API
----------
json_safe(obj)
    Recursively convert any Python/NumPy/Pandas object to a JSON-serialisable
    type.  Pass it to ``json.dump`` via the ``default`` kwarg, or use it as a
    pre-processing step before dumping.

save_json(obj, path)
    Convenience wrapper: applies ``json_safe``, creates parent directories, and
    writes a UTF-8 JSON file with 2-space indentation.

Example
-------
>>> from src.utils.json_utils import save_json
>>> save_json({"auc": np.float64(0.758), "drifted": np.bool_(True)}, Path("out.json"))
# Writes: {"auc": 0.758, "drifted": true}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(obj: Any) -> Any:
    """
    Recursively convert ``obj`` to a JSON-serialisable Python primitive.

    Handles the full set of types produced by NumPy, Pandas, and the standard
    library that the default JSON encoder cannot serialise:

    - ``np.integer``  → ``int``
    - ``np.floating`` → ``float``
    - ``np.bool_``    → ``bool``
    - ``np.ndarray``  → ``list`` (via recursion)
    - ``pd.NA``, ``pd.NaT``, ``float('nan')``, ``float('inf')`` → ``None``
    - ``pd.Timestamp`` → ISO-8601 ``str``
    - ``dict``         → recursively converted ``dict``
    - ``list | tuple | set`` → recursively converted ``list``
    - everything else  → ``str`` (last-resort fallback, preserves information)

    Parameters
    ----------
    obj:
        Any Python object.  Commonly a ``dict`` produced by a reporting
        function, but can be any nesting of the types above.

    Returns
    -------
    Any
        A JSON-serialisable equivalent of ``obj``.

    Examples
    --------
    >>> json_safe(np.float64(0.758))
    0.758
    >>> json_safe({"k": np.bool_(True), "v": [np.int64(3), pd.NA]})
    {'k': True, 'v': [3, None]}
    """
    # ── None / NA ──────────────────────────────────────────────────────────────
    if obj is None:
        return None
    if obj is pd.NA or obj is pd.NaT:
        return None

    # ── Native Python scalars (fast path) ─────────────────────────────────────
    if isinstance(obj, (str, int, float, bool)):
        # Guard against float('nan') / float('inf') which json can't encode
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return None
        return obj

    # ── NumPy scalars ─────────────────────────────────────────────────────────
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (val != val or abs(val) == float("inf")) else val

    # ── NumPy array ───────────────────────────────────────────────────────────
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]

    # ── Pandas Timestamp ──────────────────────────────────────────────────────
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # ── Mappings ──────────────────────────────────────────────────────────────
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    # ── Sequences ─────────────────────────────────────────────────────────────
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]

    # ── Last-resort fallback ──────────────────────────────────────────────────
    # Preserves the value as a readable string rather than raising an exception.
    return str(obj)


def save_json(obj: Any, path: Path) -> None:
    """
    Serialise ``obj`` to a JSON file at ``path``.

    Parent directories are created automatically.  The file is written with
    UTF-8 encoding and 2-space indentation for human readability.  All
    non-serialisable types are normalised via :func:`json_safe` before writing.

    Parameters
    ----------
    obj:
        The object to serialise.  Typically a ``dict`` containing the report
        produced by a pipeline stage.
    path:
        Destination file path.  Must be absolute or relative to the current
        working directory.  The ``.json`` extension is not enforced but
        is strongly recommended for clarity.

    Raises
    ------
    OSError
        If the file cannot be written (e.g. permissions error).

    Examples
    --------
    >>> save_json({"auc": 0.758, "n_rows": 590_540}, Path("reports/metrics.json"))
    # Creates reports/ if needed; writes a 2-space-indented JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(json_safe(obj), fh, indent=2, ensure_ascii=False)
