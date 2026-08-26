from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Optional
import numpy as np


class SlidingFairness:
    """Windowed fairness diagnostics for evolving classification streams.

    Missing/under-supported groups are represented explicitly as unavailable.
    A lack of evidence is never converted to a zero disparity.
    """

    def __init__(self, window_size: int = 500, positive_label: Any = 1, min_group_n: int = 10):
        self.buffer = deque(maxlen=max(20, int(window_size)))
        self.positive_label = positive_label
        self.min_group_n = int(min_group_n)

    def update(self, y_true, y_pred, sensitive):
        if y_pred is not None and sensitive is not None:
            self.buffer.append((y_true, y_pred, sensitive))

    def _stats(self):
        stats = defaultdict(
            lambda: {
                "n": 0,
                "correct": 0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "pred_pos": 0,
                "tp_by_label": defaultdict(int),
                "fp_by_label": defaultdict(int),
                "fn_by_label": defaultdict(int),
                "labels": set(),
            }
        )
        for yt, yp, g in self.buffer:
            s = stats[g]
            s["n"] += 1
            s["correct"] += int(yt == yp)
            s["labels"].update([yt, yp])
            if yt == yp:
                s["tp_by_label"][yt] += 1
            else:
                s["fp_by_label"][yp] += 1
                s["fn_by_label"][yt] += 1

            pos_t = yt == self.positive_label
            pos_p = yp == self.positive_label
            s["pred_pos"] += int(pos_p)
            if pos_t and pos_p:
                s["tp"] += 1
            elif not pos_t and pos_p:
                s["fp"] += 1
            elif not pos_t and not pos_p:
                s["tn"] += 1
            else:
                s["fn"] += 1
        return stats

    @staticmethod
    def _gap(vals: list[float]) -> Optional[float]:
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        return float(max(vals) - min(vals)) if len(vals) >= 2 else None

    @staticmethod
    def _macro_f1(s: dict[str, Any]) -> float:
        vals = []
        for label in s["labels"]:
            tp = s["tp_by_label"][label]
            fp = s["fp_by_label"][label]
            fn = s["fn_by_label"][label]
            denom = 2 * tp + fp + fn
            vals.append(2 * tp / denom if denom else 0.0)
        return float(np.mean(vals)) if vals else 0.0

    def compute(self) -> dict[str, Any]:
        stats = self._stats()
        valid = {g: s for g, s in stats.items() if s["n"] >= self.min_group_n}
        support = {str(g): int(s["n"]) for g, s in stats.items()}
        if len(valid) < 2:
            return {
                "status": "insufficient_group_support",
                "groups": support,
                "window_n": len(self.buffer),
                "dp_diff": None,
                "equal_opportunity_diff": None,
                "equalized_odds_gap": None,
                "predictive_parity_diff": None,
                "error_rate_gap": None,
                "worst_group_accuracy": None,
                "worst_group_macro_f1": None,
                "group_performance": {},
            }

        dp, tpr, fpr, ppv, err = [], [], [], [], []
        group_perf = {}
        for g, s in valid.items():
            n = s["n"]
            dp.append(s["pred_pos"] / n)
            tpr.append(s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else None)
            fpr.append(s["fp"] / (s["fp"] + s["tn"]) if s["fp"] + s["tn"] else None)
            ppv.append(s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else None)
            err.append((s["fp"] + s["fn"]) / n)
            group_perf[str(g)] = {
                "n": int(n),
                "accuracy": float(s["correct"] / n) if n else None,
                "macro_f1": self._macro_f1(s),
            }

        tpr_gap = self._gap(tpr)
        fpr_gap = self._gap(fpr)
        eodds = max([v for v in [tpr_gap, fpr_gap] if v is not None], default=None)
        accs = [v["accuracy"] for v in group_perf.values() if v.get("accuracy") is not None]
        f1s = [v["macro_f1"] for v in group_perf.values() if v.get("macro_f1") is not None]
        return {
            "status": "ok",
            "groups": support,
            "window_n": len(self.buffer),
            "dp_diff": self._gap(dp),
            "equal_opportunity_diff": tpr_gap,
            "equalized_odds_gap": eodds,
            "predictive_parity_diff": self._gap(ppv),
            "error_rate_gap": self._gap(err),
            "worst_group_accuracy": min(accs) if accs else None,
            "worst_group_macro_f1": min(f1s) if f1s else None,
            "group_performance": group_perf,
        }
