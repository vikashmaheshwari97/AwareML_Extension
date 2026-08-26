from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import pandas as pd


@dataclass(frozen=True)
class BuiltinStream:
    name: str
    task: str
    stream_kind: str
    description: str


BUILTIN_STREAMS = {
    "Electricity (Elec2)": BuiltinStream(
        name="Electricity (Elec2)", task="binary_classification", stream_kind="temporally_ordered",
        description="Electricity-market stream commonly used for concept-drift evaluation.",
    ),
    "Phishing": BuiltinStream(
        name="Phishing", task="binary_classification", stream_kind="ordered_as_stream",
        description="River phishing classification dataset, useful as a second real benchmark source.",
    ),
}


def load_builtin_stream(name: str, max_samples: int = 5000) -> pd.DataFrame:
    """Load a supported River dataset lazily and convert it to the common DataFrame format."""
    try:
        from river import datasets
    except Exception as e:
        raise RuntimeError("River is required for built-in benchmark streams. Install requirements.txt first.") from e

    if name == "Electricity (Elec2)":
        dataset = datasets.Elec2()
    elif name == "Phishing":
        dataset = datasets.Phishing()
    else:
        raise ValueError(f"Unknown built-in stream: {name}")

    rows = []
    for i, (x, y) in enumerate(dataset):
        if i >= int(max_samples):
            break
        row = dict(x)
        row["target"] = y
        rows.append(row)
    if not rows:
        raise RuntimeError(f"Dataset '{name}' produced no rows.")
    return pd.DataFrame(rows)
