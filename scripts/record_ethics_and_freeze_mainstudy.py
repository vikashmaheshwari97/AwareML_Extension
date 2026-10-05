from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P16 = ROOT / "data/journal/trust_calibration_phase16_v1/design/protocol.json"
P17 = ROOT / "data/journal/information_seeking_v1/design/protocol.json"


def update(path: Path, status: str, reference: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    ethics = data.setdefault("ethics", {})
    ethics["status"] = status
    ethics["reference"] = reference
    ethics["notes"] = "Institutional determination recorded from the supplied written reference."
    data["status"] = "final" if "phase" in data else "final_ready"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Record a real institutional ethics determination and freeze Main Study designs.")
    parser.add_argument("--status", required=True, choices=["approved", "exempt", "not_required"])
    parser.add_argument("--reference", required=True, help="Actual written institutional/ETIS determination reference.")
    parser.add_argument("--phase17-covered", choices=["yes", "no"], default="yes")
    args = parser.parse_args()

    reference = args.reference.strip()
    if len(reference) < 4 or "PLACEHOLDER" in reference.upper() or "ACTUAL-" in reference.upper():
        raise SystemExit("A real written determination/reference is required; do not use a fabricated placeholder.")

    update(P16, args.status, reference)
    print("[recorded] Phase-16 ethics determination")

    if args.phase17_covered == "yes":
        update(P17, args.status, reference)
        print("[recorded] Phase-17 ethics determination under the same documented scope")

    subprocess.run([sys.executable, "-m", "scripts.freeze_phase16_design"], cwd=ROOT, check=True)
    if args.phase17_covered == "yes":
        subprocess.run([sys.executable, "-m", "scripts.freeze_phase17_design"], cwd=ROOT, check=True)

    print("DESIGN FREEZE: PASS")
    print("Configure the final-collection environment variables and restart Streamlit.")


if __name__ == "__main__":
    main()
