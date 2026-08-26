from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _distribution(series: pd.Series) -> dict[str, Any]:
    counts = series.value_counts(dropna=False)
    n = int(len(series))
    rows = []
    for label, count in counts.items():
        rows.append({
            "label": _jsonable(label),
            "count": int(count),
            "fraction": (float(count) / n) if n else None,
        })
    return {
        "n": n,
        "n_unique": int(series.nunique(dropna=False)),
        "values": rows,
    }


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Stable-enough in-process dataframe fingerprint for audit metadata.

    Exact reproducibility is anchored by ``source_sha256`` whenever a source
    file is available. This dataframe digest additionally protects in-memory UI
    experiments where there may be no durable path.
    """
    h = hashlib.sha256()
    schema = [(str(c), str(df[c].dtype)) for c in df.columns]
    h.update(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(df, index=True, categorize=True).to_numpy(dtype="uint64", copy=False)
    h.update(hashed.tobytes())
    return h.hexdigest()


def build_dataset_provenance(
    df: pd.DataFrame,
    target: str,
    sensitive_attribute: Optional[str] = None,
    source_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    if target not in df.columns:
        raise ValueError("Target column %r is missing." % target)
    if sensitive_attribute and sensitive_attribute not in df.columns:
        raise ValueError("Sensitive attribute %r is missing." % sensitive_attribute)

    source_file_name = None
    source_size_bytes = None
    source_hash = None
    if source_path is not None:
        path = Path(source_path)
        if path.exists() and path.is_file():
            source_file_name = path.name
            source_size_bytes = int(path.stat().st_size)
            source_hash = file_sha256(path)

    schema = [
        {
            "name": str(c),
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=False)),
        }
        for c in df.columns
    ]
    numeric_features = [str(c) for c in df.drop(columns=[target]).select_dtypes(include=np.number).columns]
    categorical_features = [str(c) for c in df.columns if c != target and str(c) not in set(numeric_features)]

    out: dict[str, Any] = {
        "source_file_name": source_file_name,
        "source_size_bytes": source_size_bytes,
        "source_sha256": source_hash,
        "dataframe_sha256": dataframe_sha256(df),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "feature_count": int(max(0, df.shape[1] - 1)),
        "target": str(target),
        "schema": schema,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_fraction": float(df.isna().mean().mean()) if len(df.columns) else 0.0,
        "target_distribution": _distribution(df[target]),
        "sensitive_attribute": str(sensitive_attribute) if sensitive_attribute else None,
        "sensitive_distribution": (
            _distribution(df[sensitive_attribute]) if sensitive_attribute else None
        ),
    }
    return out
