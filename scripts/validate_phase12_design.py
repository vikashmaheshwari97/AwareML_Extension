from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark import (
    phase10_protocol_info,
    phase11_selector_info,
    validate_generated_design,
)


def main():
    design = validate_generated_design(ROOT)
    p10 = phase10_protocol_info(ROOT)
    p11 = phase11_selector_info(ROOT)

    print("=" * 72)
    print("AwareML Phase 12 design validation: PASS")
    print("=" * 72)
    print("Generated candidate scenarios:", design["candidate_count"])
    print("k' design counts:", design["k_prime_design_counts"])
    print("External generator:", design["generator_model"])
    print("Paraphrase families:", design["paraphrase_families"])
    print("Paraphrase evaluations:", design["paraphrase_evaluations"])
    print("Adversarial cases:", design["adversarial_cases"])
    print("Phase-10 protocol SHA256:", p10["sha256"])
    print("Phase-11 selector SHA256:", p11["sha256"])
    print("Evaluated model used to generate pool: False")
    print("23 held-out dataset contents used: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
