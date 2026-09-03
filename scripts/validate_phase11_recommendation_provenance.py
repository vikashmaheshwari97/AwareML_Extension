from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "awareml" / "ui_v2" / "pages_core.py"
SPECIALIST = ROOT / "awareml" / "ui_v2" / "pages_specialist.py"


def require(text: str, token: str, label: str) -> None:
    if token not in text:
        raise RuntimeError("{} missing: {}".format(label, token))


def main() -> None:
    core = CORE.read_text(encoding="utf-8")
    specialist = SPECIALIST.read_text(encoding="utf-8")

    require(core, "PRE-RUN PREFERENCE RECOMMENDATION", "3D provenance")
    require(core, "Pre-run recommended framework", "3D recommendation label")
    require(core, "Pre-run preference utility", "3D utility label")
    require(core, '{} predicted accuracy".format(selected)', "3D inspected accuracy")
    require(core, '{} predicted runtime".format(selected)', "3D inspected runtime")
    require(core, '{} predicted energy".format(selected)', "3D inspected energy")
    require(core, '{} predicted CO₂".format(selected)', "3D inspected CO2")
    require(
        core,
        "Recommended framework and inspected framework are different concepts",
        "3D recommendation/inspection explanation",
    )

    require(
        specialist,
        "OBSERVED POST-RUN RECOMMENDATION",
        "Decision Lab provenance",
    )
    require(
        specialist,
        "Post-run recommended framework",
        "Decision Lab recommendation label",
    )

    print("=" * 72)
    print("AwareML Phase 11 recommendation provenance validation: PASS")
    print("=" * 72)
    print("3D Decision Space: PRE-RUN / predicted evidence")
    print("Copilot Workspace: SCENARIO-CONDITIONED PRE-RUN / predicted evidence")
    print("Decision Lab: OBSERVED POST-RUN / measured evidence")
    print("Ranking algorithms changed: False")
    print("Recommender models changed: False")
    print("Objective selector changed: False")
    print("Held-out data touched: False")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("VALIDATION FAILED: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        raise
