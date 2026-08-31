from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.protocol import (
    validate_frozen_protocol,
    validate_static_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate Phase-10 inputs before freezing; does not require Ollama.",
    )
    parser.add_argument(
        "--skip-live-ollama-match",
        action="store_true",
        help="Validate frozen files without rechecking current Ollama version/model digest.",
    )
    parser.add_argument("--ollama-base-url", default=None)
    args = parser.parse_args()

    if args.preflight:
        result = validate_static_inputs(ROOT, verify_git=True)
        print("=" * 72)
        print("AwareML Phase 10 preflight validation: PASS")
        print("=" * 72)
        print("Engineering baseline:", result["engineering_baseline"]["commit"])
        print("Objective vocabulary:", result["objectives"]["display_labels"])
        heldout = result["dataset_policy"]["canonical_role_resolution"]["final_heldout_evaluation"]
        print("23-dataset role: FINAL HELD-OUT EVALUATION")
        print("Held-out identity status:", heldout["identity_status"])
        print("Held-out contents read: False")
        print("=" * 72)
        return

    result = validate_frozen_protocol(
        ROOT,
        require_ollama_match=not args.skip_live_ollama_match,
        base_url_override=args.ollama_base_url,
    )
    print("=" * 72)
    print("AwareML Phase 10 COMPLETE validation: PASS")
    print("=" * 72)
    print("Protocol:", result["protocol_path"])
    print("SHA256:", result["sha256"])
    print("Engineering baseline:", result["engineering_baseline_commit"])
    print("Objective vocabulary:", result["objectives"])
    print("Journal LLM:", result["journal_model"])
    print("Ollama version:", result["ollama_version"])
    print("Held-out expected count:", result["heldout_expected_count"])
    print("Held-out identity status:", result["heldout_identity_status"])
    print("Held-out dataset contents read by Phase 10: False")
    print("Release status: frozen")
    print("=" * 72)


if __name__ == "__main__":
    main()
