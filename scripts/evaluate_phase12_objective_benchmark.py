from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import evaluate_main_benchmark


def main():
    m = evaluate_main_benchmark(ROOT)
    print("=" * 72)
    print("AwareML Phase 12 objective-selection metrics")
    print("=" * 72)
    print("N:", m["n"])
    for obj, row in m["per_objective"].items():
        print(
            "{:8s} precision={:.4f} recall={:.4f} f1={:.4f}".format(
                obj, row["precision"], row["recall"], row["f1"]
            )
        )
    print("Micro-F1:", "{:.4f}".format(m["micro_f1"]))
    print("Macro-F1:", "{:.4f}".format(m["macro_f1"]))
    print("Exact-match rate:", "{:.4f}".format(m["exact_match_rate"]))
    print("By k':")
    for k, row in m["by_k_prime"].items():
        print(
            "  k'={} n={} exact={:.4f} jaccard={:.4f}".format(
                k, row["n"], row["exact_match_rate"], row["mean_jaccard"]
            )
        )
    print("Prediction status counts:", m["prediction_status_counts"])
    print("=" * 72)


if __name__ == "__main__":
    main()
