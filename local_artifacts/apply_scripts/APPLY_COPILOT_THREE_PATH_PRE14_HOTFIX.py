from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
PAYLOAD = ROOT / "three_path_payload"
BACKUP = ROOT / ".pre14_three_path_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")

def sha256(path: Path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def protected():
    items = [
        ROOT / "data/journal/objective_selection_benchmark_v1/frozen/manifest.json",
        ROOT / "data/journal/recommender_multiobjective_validation_v1/frozen/manifest.json",
        ROOT / "awareml/llm/objective_selection_v31.py",
        ROOT / "awareml/llm/objective_selection_v3.py",
        ROOT / "awareml/recommender/v2_service.py",
        ROOT / "data/meta/models/recommender_v2/manifest.json",
        ROOT / "data/meta/snapshots/meta_logs_v2.json",
        ROOT / "data/meta/snapshots/recommender_train_v2.parquet",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in items if p.exists()}

def backup(path: Path):
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

def patch_pages():
    path = ROOT / "awareml/ui_v2/pages_copilot.py"
    if not path.exists():
        raise RuntimeError("Missing awareml/ui_v2/pages_copilot.py")
    backup(path)
    text = path.read_text(encoding="utf-8")

    import_line = "from .copilot_three_path import render_historical_preference_prior_tab, render_dataset_aware_v2_tab"
    if import_line not in text:
        anchor = "from .state import ROOT, ensure_research_state"
        if anchor not in text:
            raise RuntimeError("Could not locate pages_copilot import anchor.")
        text = text.replace(anchor, anchor + "\n" + import_line, 1)

    marker = "# PRE14_HISTORICAL_META_RECOMMENDER_WRAPPER"
    marker2 = "# PRE14_THREE_PATH_COPILOT_WRAPPER"
    if marker in text:
        prefix = text.split(marker, 1)[0].rstrip() + "\n\n"
    elif marker2 in text:
        prefix = text.split(marker2, 1)[0].rstrip() + "\n\n"
    else:
        raise RuntimeError("Could not locate Copilot tab-wrapper marker.")

    wrapper = """# PRE14_THREE_PATH_COPILOT_WRAPPER
def copilot_workspace_page():
    st.markdown("## AwareML Copilot · choose the evidence path that fits your situation")
    st.caption(
        "Goal Copilot understands what you want. Historical Preference Prior shows what generally worked before. "
        "Dataset-aware ML Recommender V2 predicts what should work for the dataset you actually loaded."
    )

    overview = st.columns(3)
    with overview[0]:
        st.markdown("**1 · Goal Copilot**")
        st.caption("No dataset · LLaMA 3 8B + V3.1 · understands Accuracy / Runtime / Energy / CO2 · not a framework selector by itself.")
    with overview[1]:
        st.markdown("**2 · Historical Preference Prior**")
        st.caption("No dataset · 705 runs summarized · global starting point · historical aggregation, not machine learning.")
    with overview[2]:
        st.markdown("**3 · Dataset-aware ML Recommender V2**")
        st.caption("Dataset + target · learned models · dataset-specific pre-run framework prediction · true ML recommender.")

    goal_tab, prior_tab, v2_tab = st.tabs([
        "1 · Goal Copilot",
        "2 · Historical Preference Prior · No dataset",
        "3 · Dataset-aware ML Recommender V2",
    ])
    with goal_tab:
        _goal_copilot_workspace_page()
    with prior_tab:
        render_historical_preference_prior_tab()
    with v2_tab:
        render_dataset_aware_v2_tab()
"""
    path.write_text(prefix + wrapper, encoding="utf-8")

def main():
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError("Run this installer from the AwareML_Extension project root.")
    if not PAYLOAD.exists():
        raise RuntimeError("Missing three_path_payload. Extract the complete ZIP first.")

    before = protected()
    BACKUP.mkdir(parents=True, exist_ok=True)
    copy_payload()
    patch_pages()
    after = protected()

    changed = [k for k in before if before.get(k) != after.get(k)]
    if changed:
        raise RuntimeError("Protected frozen/model evidence changed unexpectedly: {}".format(changed))

    print("=" * 92)
    print("AwareML THREE-PATH Copilot / pre-Phase-14 hotfix: APPLIED")
    print("=" * 92)
    print("Backup:", BACKUP)
    print("Protected frozen/model evidence unchanged: True")
    print()
    print("Run next:")
    print(r"  pytest -q .\tests\test_copilot_three_path_pre14.py")
    print(r"  python -m scripts.validate_copilot_three_path_pre14")
    print(r"  python -m scripts.audit_meta_logs_v2_pre14")
    print(r"  streamlit run app.py")
    print()
    print("Do not git-add .pre14_three_path_backup or three_path_payload")
    print("=" * 92)

if __name__ == "__main__":
    main()
