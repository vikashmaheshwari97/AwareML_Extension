from __future__ import annotations

from dataclasses import dataclass, asdict
import random
from typing import Any


@dataclass
class TrustCase:
    condition: str
    shown_framework: str
    oracle_framework: str
    reliability: str
    explanation: str
    utility_shown: float
    utility_oracle: float

    def to_dict(self):
        return asdict(self)


class TrustCalibrationStudy:
    """Creates matched-style correct/weak/wrong conditions from observed ranking utilities."""

    TEMPLATE = (
        "The system recommends **{framework}** because it provides a balanced result across the configured "
        "objectives. Its weighted utility is {utility:.3f}. The recommendation should be checked against "
        "the comparison evidence before use."
    )

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def build_case(self, ranking_rows: list[dict[str, Any]], condition: str | None = None) -> TrustCase:
        rows = sorted(ranking_rows, key=lambda r: float(r["utility"]), reverse=True)
        if len(rows) < 3:
            raise ValueError("Trust calibration requires at least three ranked framework candidates.")
        condition = condition or self.rng.choice(["correct", "weak", "wrong"])
        oracle = rows[0]
        if condition == "correct":
            shown = rows[0]
            reliability = "highest-observed-utility"
        elif condition == "weak":
            shown = rows[1]
            reliability = "second-best-observed-utility"
        elif condition == "wrong":
            shown = rows[-1]
            reliability = "lowest-observed-utility"
        else:
            raise ValueError("Condition must be correct, weak, or wrong.")
        return TrustCase(
            condition=condition,
            shown_framework=str(shown["framework"]),
            oracle_framework=str(oracle["framework"]),
            reliability=reliability,
            explanation=self.TEMPLATE.format(framework=shown["framework"], utility=float(shown["utility"])),
            utility_shown=float(shown["utility"]),
            utility_oracle=float(oracle["utility"]),
        )


def calibration_metrics(responses):
    """Compute simple discrimination/overtrust summaries from exported responses."""
    import pandas as pd
    df = pd.DataFrame(responses)
    if df.empty:
        return {}
    out = {}
    if {"condition", "trust"}.issubset(df.columns):
        out["mean_trust_by_condition"] = df.groupby("condition")["trust"].mean().to_dict()
    if {"condition", "accepted"}.issubset(df.columns):
        wrong = df[df["condition"] == "wrong"]
        out["wrong_acceptance_rate"] = float(wrong["accepted"].astype(float).mean()) if len(wrong) else None
    return out
