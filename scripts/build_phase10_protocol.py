from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.protocol import build_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-frozen preview of Journal Experimental Protocol v1."
    )
    parser.add_argument(
        "--verify-ollama",
        action="store_true",
        help="Also require and record the exact llama3:8b Ollama runtime.",
    )
    parser.add_argument("--ollama-base-url", default=None)
    args = parser.parse_args()

    protocol = build_protocol(
        ROOT,
        require_ollama=args.verify_ollama,
        base_url_override=args.ollama_base_url,
    )

    out = ROOT / "artifacts" / "phase10" / "journal_protocol_preview.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("AwareML Phase 10 protocol preview: PASS")
    print("=" * 72)
    print("Engineering baseline:", protocol["engineering_baseline"]["commit"])
    print("Objectives:", protocol["objective_vocabulary"]["display_labels"])
    heldout = protocol["dataset_split_policy"]["canonical_role_resolution"]["final_heldout_evaluation"]
    print("Held-out role: FINAL EVALUATION")
    print("Held-out expected count:", heldout["expected_count"])
    print("Held-out identity status:", heldout["identity_status"])
    print("Journal LLM:", protocol["journal_llm"]["required_model_tag"])
    print("Runtime verified:", protocol["journal_llm"]["runtime_verified"])
    print("Preview:", out)
    print("=" * 72)


if __name__ == "__main__":
    main()
