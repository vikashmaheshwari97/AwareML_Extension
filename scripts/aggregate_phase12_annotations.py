from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import aggregate_annotations


def main():
    result = aggregate_annotations(ROOT)
    agreement = result["agreement"]

    print("=" * 72)
    print("AwareML Phase 12 human annotation aggregation: COMPLETE")
    print("=" * 72)
    print("Annotators:", agreement["annotator_count"])
    print("Annotator IDs:", agreement["annotator_ids"])
    print("Annotated scenarios:", agreement["scenario_count"])
    print("Resolved ground truth:", agreement["resolved_ground_truth"])
    print("Hard/ambiguous cases:", agreement["hard_cases"])
    print("Per-objective Fleiss kappa:")
    for obj, value in agreement["per_objective_fleiss_kappa"].items():
        print("  {:8s} {}".format(obj, "undefined" if value is None else "{:.4f}".format(value)))
    print("Mean defined kappa:", agreement["mean_defined_fleiss_kappa"])
    print("Unanimous full-set agreement:", agreement["unanimous_full_set_agreement_rate"])
    print("Mean pairwise Jaccard:", agreement["mean_pairwise_jaccard"])
    print("=" * 72)


if __name__ == "__main__":
    main()
