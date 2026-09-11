from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path.cwd()
PAYLOAD = ROOT / "payload"
BACKUP = (
    ROOT
    / ".phase14_integrated_labs_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def sha256(path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes():
    paths = [
        ROOT / "data/journal/objective_selection_benchmark_v1/frozen/manifest.json",
        ROOT / "data/journal/recommender_multiobjective_validation_v1/frozen/manifest.json",
        ROOT / "awareml/llm/objective_selection_v3.py",
        ROOT / "awareml/llm/objective_selection_v31.py",
        ROOT / "awareml/recommender/v2_service.py",
        ROOT / "awareml/recommender/v2_ranking.py",
        ROOT / "data/meta/models/recommender_v2/manifest.json",
    ]
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in paths
        if path.exists()
    }


def backup(path):
    if not path.exists():
        return
    out = BACKUP / path.relative_to(ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)


def copy_payload():
    for src in PAYLOAD.rglob("*"):
        if not src.is_file():
            continue
        dest = ROOT / src.relative_to(PAYLOAD)
        backup(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def patch_advanced():
    path = ROOT / "awareml" / "ui_v2" / "pages_advanced.py"
    text = path.read_text(encoding="utf-8")
    original = text

    text = text.replace(
        "from .phase14_hardening import phase14_hardening_page\n",
        "",
    )
    text = text.replace(
        '        "Phase 14 · Fairness + Sustainability Hardening": phase14_hardening_page,\n',
        "",
    )

    old_desc = (
        '"Phase 9.5 upgrades the specialist views while keeping the original '
        'paper-baseline recommender and protocol available for historical comparison."'
    )
    new_desc = (
        '"Specialist research workspaces integrate the current journal-extension '
        'analyses while keeping the original paper-baseline recommender and '
        'protocol available for historical comparison."'
    )
    text = text.replace(old_desc, new_desc)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print("Patched: Advanced Research Labs selector")
    else:
        print("Advanced Research Labs already integrated.")


def add_import_once(text):
    import_line = (
        "from .phase14_integrated_sections import (\n"
        "    render_phase14_fairness_details,\n"
        "    render_phase14_sustainability_details,\n"
        ")\n"
    )
    if "render_phase14_fairness_details" in text:
        return text

    anchor = (
        "from .study_labs_v3 import (\n"
        "    information_seeking_research_page,\n"
        "    trust_calibration_research_page,\n"
        ")\n"
    )
    if anchor not in text:
        raise RuntimeError("Could not locate pages_specialist import anchor.")
    return text.replace(anchor, anchor + import_line, 1)


def patch_specialist():
    path = ROOT / "awareml" / "ui_v2" / "pages_specialist.py"
    text = path.read_text(encoding="utf-8")
    original = text
    text = add_import_once(text)

    fairness_call = "    render_phase14_fairness_details(results)\n"
    if fairness_call not in text:
        anchor = "\n    if not common_labels:\n"
        if anchor not in text:
            raise RuntimeError(
                "Could not locate Fairness Lab Phase-14 integration anchor."
            )
        text = text.replace(
            anchor,
            "\n" + fairness_call + anchor,
            1,
        )

    sustain_call = "    render_phase14_sustainability_details(results)\n"
    if sustain_call not in text:
        anchor = (
            '        st.caption("ρ≈1 means energy and CO₂ rank frameworks almost '
            'identically; weighting both heavily can double-count the same efficiency signal.")\n'
        )
        if anchor not in text:
            raise RuntimeError(
                "Could not locate Sustainability Lab Phase-14 integration anchor."
            )
        text = text.replace(
            anchor,
            anchor + "\n" + sustain_call,
            1,
        )

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print("Patched: Fairness Lab + Sustainability Lab")
    else:
        print("Specialist labs already integrated.")


def remove_obsolete_page():
    path = ROOT / "awareml" / "ui_v2" / "phase14_hardening.py"
    if path.exists():
        backup(path)
        path.unlink()
        print("Removed obsolete separate Phase-14 workspace module.")


def main():
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )
    if not PAYLOAD.exists():
        raise RuntimeError("payload folder is missing. Extract the complete ZIP.")

    before = protected_hashes()
    BACKUP.mkdir(parents=True, exist_ok=True)

    copy_payload()
    patch_advanced()
    patch_specialist()
    remove_obsolete_page()

    after = protected_hashes()
    changed = [
        key for key in before
        if before.get(key) != after.get(key)
    ]
    if changed:
        raise RuntimeError(
            "Protected research/model artifacts changed unexpectedly: {}".format(changed)
        )

    print("=" * 96)
    print("AwareML Phase-14 Integrated Labs Hotfix: APPLIED")
    print("=" * 96)
    print("Backup:", BACKUP)
    print("Separate Phase-14 workspace: REMOVED")
    print("Calibration fairness: integrated into Fairness Lab")
    print("Sustainability protocol + repeatability: integrated into Sustainability Lab")
    print("Dutch Census Downloads resolver: INSTALLED")
    print("Protected Phase-12/13/V3/V3.1/Recommender-V2 artifacts unchanged: True")
    print()
    print("Run:")
    print(r"  pytest -q .\tests\test_phase14_hardening.py .\tests\test_phase14_integrated_labs.py")
    print(r"  python -m scripts.validate_phase14_hardening")
    print(r"  streamlit run app.py")
    print("=" * 96)


if __name__ == "__main__":
    main()
