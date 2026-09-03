from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(script, *args):
    cmd = [PYTHON, str(ROOT / "scripts" / script)] + list(args)
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    # This completion script intentionally begins AFTER human realism filtering,
    # independent annotation, and paraphrase review have been completed.
    run("aggregate_phase12_annotations.py")
    run("freeze_phase12_ground_truth.py")
    run("run_phase12_objective_benchmark.py")
    run("evaluate_phase12_objective_benchmark.py")
    run("run_phase12_paraphrase_benchmark.py")
    run("run_phase12_adversarial_benchmark.py")
    run("freeze_phase12_benchmark.py")
    run("validate_phase12_complete.py")

    print("\n" + "=" * 72)
    print("ALL PHASE 12 STEPS PASSED")
    print("objective_selection_benchmark_v1 is frozen.")
    print("=" * 72)


if __name__ == "__main__":
    main()
