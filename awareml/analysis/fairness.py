from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any, Mapping, Optional

import numpy as np


class SlidingFairness:
    """Windowed classification + calibration fairness for evolving streams."""

    def __init__(
        self,
        window_size: int = 500,
        positive_label: Any = 1,
        min_group_n: int = 10,
        calibration_bins: int = 10,
        degenerate_prediction_threshold: float = 0.95,
    ):
        self.buffer = deque(maxlen=max(20, int(window_size)))
        self.positive_label = positive_label
        self.min_group_n = int(min_group_n)
        self.calibration_bins = max(2, int(calibration_bins))
        self.degenerate_prediction_threshold = min(
            1.0, max(0.5, float(degenerate_prediction_threshold))
        )

    @staticmethod
    def _finite_probability(value: Any) -> Optional[float]:
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            return None
        return value

    def _positive_probability(self, y_proba: Any) -> Optional[float]:
        if not isinstance(y_proba, Mapping) or not y_proba:
            return None
        if self.positive_label in y_proba:
            return self._finite_probability(y_proba.get(self.positive_label))
        matches = [
            value
            for key, value in y_proba.items()
            if str(key) == str(self.positive_label)
        ]
        if len(matches) == 1:
            return self._finite_probability(matches[0])
        return None

    def update(self, y_true, y_pred, sensitive, y_proba=None):
        if sensitive is None:
            return
        positive_probability = self._positive_probability(y_proba)
        if y_pred is not None or positive_probability is not None:
            self.buffer.append((y_true, y_pred, sensitive, positive_probability))

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
        for item in self.buffer:
            yt, yp, group = item[:3]
            if yp is None:
                continue
            s = stats[group]
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
    def _gap(vals):
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        return float(max(vals) - min(vals)) if len(vals) >= 2 else None

    @staticmethod
    def _macro_f1(s):
        vals = []
        for label in s["labels"]:
            tp = s["tp_by_label"][label]
            fp = s["fp_by_label"][label]
            fn = s["fn_by_label"][label]
            denom = 2 * tp + fp + fn
            vals.append(2 * tp / denom if denom else 0.0)
        return float(np.mean(vals)) if vals else 0.0

    def _prediction_diagnostics(self):
        hard = [
            item[1]
            for item in self.buffer
            if len(item) >= 2 and item[1] is not None
        ]
        probabilities = [
            self._finite_probability(item[3])
            for item in self.buffer
            if len(item) >= 4
        ]
        probabilities = [p for p in probabilities if p is not None]

        if hard:
            counts = Counter(hard)
            majority_label, majority_count = counts.most_common(1)[0]
            majority_fraction = float(majority_count / len(hard))
            class_count = int(len(counts))
            if class_count == 1:
                hard_status = "constant"
            elif majority_fraction >= self.degenerate_prediction_threshold:
                hard_status = "near_constant"
            else:
                hard_status = "ok"
            positive_fraction = float(
                sum(pred == self.positive_label for pred in hard) / len(hard)
            )
        else:
            majority_label = None
            majority_fraction = None
            class_count = 0
            hard_status = "unavailable"
            positive_fraction = None

        if probabilities:
            p = np.asarray(probabilities, dtype=float)
            probability_std = float(np.std(p))
            probability_range = float(np.max(p) - np.min(p))
            if probability_range <= 1e-12:
                probability_status = "constant"
            elif probability_std <= 1e-4:
                probability_status = "near_constant"
            else:
                probability_status = "ok"
        else:
            probability_std = None
            probability_range = None
            probability_status = "unavailable"

        notes = []
        if hard_status in {"constant", "near_constant"}:
            notes.append(
                "Hard predictions are {} (majority share {:.1%}). "
                "Zero demographic-parity/equal-opportunity/equalized-odds gaps "
                "can therefore be structurally trivial and must not be interpreted "
                "as proof of a useful fair classifier.".format(
                    hard_status.replace("_", " "),
                    majority_fraction or 0.0,
                )
            )
        if probability_status in {"constant", "near_constant"}:
            notes.append(
                "Positive-class probabilities are {}. A zero Group Brier-score gap "
                "can occur because groups receive almost the same probability; "
                "inspect Group ECE and predictive performance as well.".format(
                    probability_status.replace("_", " ")
                )
            )

        return {
            "prediction_behavior_status": hard_status,
            "prediction_majority_label": majority_label,
            "prediction_majority_fraction": majority_fraction,
            "predicted_positive_fraction": positive_fraction,
            "predicted_class_count": class_count,
            "probability_behavior_status": probability_status,
            "positive_probability_std": probability_std,
            "positive_probability_range": probability_range,
            "fairness_interpretation_warning": " ".join(notes) if notes else None,
        }

    def _ece(self, y_binary, probabilities):
        if len(probabilities) == 0:
            return None
        edges = np.linspace(0.0, 1.0, self.calibration_bins + 1)
        bins = np.minimum(
            np.searchsorted(edges, probabilities, side="right") - 1,
            self.calibration_bins - 1,
        )
        bins = np.maximum(bins, 0)
        ece = 0.0
        n = float(len(probabilities))
        for idx in range(self.calibration_bins):
            mask = bins == idx
            if not np.any(mask):
                continue
            weight = float(np.sum(mask)) / n
            confidence = float(np.mean(probabilities[mask]))
            observed = float(np.mean(y_binary[mask]))
            ece += weight * abs(confidence - observed)
        return float(ece)

    def _calibration(self):
        by_group = defaultdict(lambda: {"y": [], "p": []})
        probability_rows = 0
        for item in self.buffer:
            if len(item) < 4:
                continue
            yt, _, group, probability = item
            probability = self._finite_probability(probability)
            if probability is None:
                continue
            probability_rows += 1
            by_group[group]["y"].append(
                1.0 if yt == self.positive_label else 0.0
            )
            by_group[group]["p"].append(probability)

        denominator = max(1, len(self.buffer))
        coverage = float(probability_rows / denominator)
        support = {str(g): len(v["p"]) for g, v in by_group.items()}

        if probability_rows == 0:
            return {
                "calibration_status": "unavailable",
                "calibration_reason": (
                    "No valid positive-class probabilities were exposed in this window."
                ),
                "calibration_bins": self.calibration_bins,
                "probability_coverage": 0.0,
                "calibration_group_support": support,
                "group_brier_score_gap": None,
                "group_ece_gap": None,
                "group_calibration": {},
            }

        valid = {
            group: values
            for group, values in by_group.items()
            if len(values["p"]) >= self.min_group_n
        }
        if len(valid) < 2:
            return {
                "calibration_status": "insufficient_group_support",
                "calibration_reason": (
                    "Valid probabilities exist, but fewer than two groups meet "
                    "the minimum calibration support."
                ),
                "calibration_bins": self.calibration_bins,
                "probability_coverage": coverage,
                "calibration_group_support": support,
                "group_brier_score_gap": None,
                "group_ece_gap": None,
                "group_calibration": {},
            }

        briers, eces = [], []
        group_calibration = {}
        for group, values in valid.items():
            y = np.asarray(values["y"], dtype=float)
            p = np.asarray(values["p"], dtype=float)
            brier = float(np.mean((p - y) ** 2))
            ece = self._ece(y, p)
            briers.append(brier)
            eces.append(ece)
            group_calibration[str(group)] = {
                "n": int(len(p)),
                "brier_score": brier,
                "ece": ece,
                "mean_predicted_positive_probability": float(np.mean(p)),
                "observed_positive_rate": float(np.mean(y)),
            }

        return {
            "calibration_status": "ok",
            "calibration_reason": None,
            "calibration_bins": self.calibration_bins,
            "probability_coverage": coverage,
            "calibration_group_support": support,
            "group_brier_score_gap": self._gap(briers),
            "group_ece_gap": self._gap(eces),
            "group_calibration": group_calibration,
        }

    def compute(self):
        calibration = self._calibration()
        prediction_diagnostics = self._prediction_diagnostics()
        stats = self._stats()
        valid = {
            group: s for group, s in stats.items()
            if s["n"] >= self.min_group_n
        }
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
                **prediction_diagnostics,
                **calibration,
            }

        dp, tpr, fpr, ppv, err = [], [], [], [], []
        group_perf = {}
        for group, s in valid.items():
            n = s["n"]
            predicted_positive_rate = s["pred_pos"] / n
            group_tpr = (
                s["tp"] / (s["tp"] + s["fn"])
                if s["tp"] + s["fn"] else None
            )
            group_fpr = (
                s["fp"] / (s["fp"] + s["tn"])
                if s["fp"] + s["tn"] else None
            )
            group_ppv = (
                s["tp"] / (s["tp"] + s["fp"])
                if s["tp"] + s["fp"] else None
            )
            group_err = (s["fp"] + s["fn"]) / n

            dp.append(predicted_positive_rate)
            tpr.append(group_tpr)
            fpr.append(group_fpr)
            ppv.append(group_ppv)
            err.append(group_err)

            group_perf[str(group)] = {
                "n": int(n),
                "accuracy": float(s["correct"] / n) if n else None,
                "macro_f1": self._macro_f1(s),
                "predicted_positive_rate": float(predicted_positive_rate),
                "true_positive_rate": (
                    float(group_tpr) if group_tpr is not None else None
                ),
                "false_positive_rate": (
                    float(group_fpr) if group_fpr is not None else None
                ),
                "positive_predictive_value": (
                    float(group_ppv) if group_ppv is not None else None
                ),
                "error_rate": float(group_err),
            }

        tpr_gap = self._gap(tpr)
        fpr_gap = self._gap(fpr)
        eodds = max(
            [v for v in [tpr_gap, fpr_gap] if v is not None],
            default=None,
        )
        accs = [
            v["accuracy"] for v in group_perf.values()
            if v.get("accuracy") is not None
        ]
        f1s = [
            v["macro_f1"] for v in group_perf.values()
            if v.get("macro_f1") is not None
        ]

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
            **prediction_diagnostics,
            **calibration,
        }
