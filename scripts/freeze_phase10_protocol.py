from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.protocol import freeze_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-base-url", default=None)
    args = parser.parse_args()

    result = freeze_protocol(ROOT, base_url_override=args.ollama_base_url)
    protocol = result["protocol"]
    heldout = (
        protocol["dataset_split_policy"]
        ["canonical_role_resolution"]
        ["final_heldout_evaluation"]
    )

    print("=" * 72)
    print("AwareML Phase 10 freeze: SUCCESS")
    print("=" * 72)
    print("Protocol:", result["protocol_path"])
    print("SHA256:", result["sha256"])
    print("Active marker:", result["active_protocol_path"])
    print("Journal model:", protocol["journal_llm"]["required_model_tag"])
    print("Journal model digest:", protocol["journal_llm"]["runtime_lock"]["model_digest"])
    print("Ollama version:", protocol["journal_llm"]["runtime_lock"]["ollama_version"])
    print("Objectives:", protocol["objective_vocabulary"]["display_labels"])
    print("Held-out expected count:", heldout["expected_count"])
    print("Held-out identity status:", heldout["identity_status"])
    print("Held-out contents touched: False")
    print("=" * 72)


if __name__ == "__main__":
    main()
