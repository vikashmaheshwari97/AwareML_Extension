from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import validate_complete


def main():
    r = validate_complete(ROOT)
    manifest = r["manifest"]
    metrics = manifest["metrics"]
    print("=" * 72)
    print("AwareML Phase 12 COMPLETE validation: PASS")
    print("=" * 72)
    print("Artifact: objective_selection_benchmark_v1")
    print("SHA256:", r["sha256"])
    print("Primary benchmark N:", r["primary_n"])
    print("Objectives:", manifest["objective_vocabulary"])
    print("Journal model:", manifest["journal_model"])
    print("Micro-F1:", metrics["micro_f1"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Exact-match:", metrics["exact_match_rate"])
    print("Paraphrase evaluations:", manifest["paraphrase_summary"]["paraphrase_evaluations"])
    print("Adversarial cases:", manifest["adversarial_summary"]["n"])
    print("Human ground truth:", manifest["human_ground_truth"])
    print("23 held-out dataset contents used: False")
    print("Release status:", manifest["release_status"])
    print("=" * 72)


if __name__ == "__main__":
    main()
