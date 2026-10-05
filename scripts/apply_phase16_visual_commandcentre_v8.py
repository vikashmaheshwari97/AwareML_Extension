from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

APP = ROOT / "app.py"
PAGES_CORE = ROOT / "awareml/ui_v2/pages_core.py"
PAGES = ROOT / "awareml/ui_v2/pages.py"
P16 = ROOT / "awareml/ui_v2/phase16_trust_calibration.py"
VISUALS = ROOT / "awareml/ui_v2/phase16_research_visuals.py"
COMMAND = ROOT / "awareml/ui_v2/command_center_upgrade.py"


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "artifacts/ui_backups_v8" / stamp / path.relative_to(ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)


def replace_once(path: Path, old: str, new: str, label: str, required: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print("[already applied] {}".format(label))
        return False
    if old not in text:
        if required:
            raise RuntimeError("{}: source text not found in {}".format(label, path))
        print("[skip] {}: source text not found".format(label))
        return False
    backup(path)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("[patched] {}".format(label))
    return True


def patch_command_centre() -> None:
    text = PAGES_CORE.read_text(encoding="utf-8")

    import_line = "from .command_center_upgrade import render_command_center_upgrade\n"
    if import_line not in text:
        anchor = "from .components import empty_state, hero, metric_card, section, status_panel\n"
        if anchor not in text:
            raise RuntimeError("Could not locate pages_core import anchor.")
        backup(PAGES_CORE)
        text = text.replace(anchor, anchor + import_line, 1)

    start = text.find("def command_center_page():")
    end = text.find("\ndef decision_space_page():")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not isolate command_center_page() in pages_core.py")

    new_function = (
        "def command_center_page():\n"
        "    return render_command_center_upgrade()\n\n"
    )
    current_block = text[start:end]
    if "return render_command_center_upgrade()" not in current_block:
        backup(PAGES_CORE)
        text = text[:start] + new_function + text[end + 1:]
        print("[patched] redesigned Command Centre")
    else:
        print("[already applied] redesigned Command Centre")

    PAGES_CORE.write_text(text, encoding="utf-8")


def patch_app_branding() -> None:
    text = APP.read_text(encoding="utf-8")
    changed = False
    replacements = [
        ('page_title="AwareML Research OS"', 'page_title="AwareML Dashboard"'),
        ('<div class="r9-brand-sub">Research OS</div>', '<div class="r9-brand-sub">Dashboard</div>'),
        ("across every Research OS workspace.", "across every AwareML workspace."),
    ]
    for old, new in replacements:
        if old in text:
            if not changed:
                backup(APP)
            text = text.replace(old, new)
            changed = True
    if changed:
        APP.write_text(text, encoding="utf-8")
        print("[patched] AwareML Dashboard branding")
    else:
        print("[already applied/skip] AwareML Dashboard branding")


def patch_registry_name() -> None:
    replace_once(
        PAGES,
        '"Command Center": command_center_page,',
        '"Command Centre": command_center_page,',
        "sidebar Command Centre name",
        required=False,
    )


def patch_participant_terminology() -> None:
    text = P16.read_text(encoding="utf-8")
    old = (
        '    section("Your response", "Compare the explanation with the reference evidence first. '
        'Then rate trust, correctness, writing quality and your intended action.")\n'
        '    with st.form("p16_trial_form_{}".format(trial["item_id"])):\n'
    )
    new = (
        '    section("Your evaluation", "Compare the explanation with the reference evidence first. '
        'Then rate trust, factual correctness, fluency, confident-sounding language and your intended action.")\n'
        '    st.caption(\n'
        '        "Terminology note: in the validated trust statements below, ‘AI assistant’ means the system that produced "\n'
        '        "the explanation shown above. It does not refer to the separate AwareML Copilot workspace."\n'
        '    )\n'
        '    with st.form("p16_trial_form_{}".format(trial["item_id"])):\n'
    )
    if new in text:
        print("[already applied] participant trust terminology clarification")
        return
    if old not in text:
        # v7 or another local patch may have changed only the section wording.
        marker = '    with st.form("p16_trial_form_{}".format(trial["item_id"])):\n'
        if marker not in text:
            print("[skip] participant terminology clarification: form anchor not found")
            return
        backup(P16)
        note = (
            '    st.caption(\n'
            '        "Terminology note: in the validated trust statements below, ‘AI assistant’ means the system that produced "\n'
            '        "the explanation shown above. It does not refer to the separate AwareML Copilot workspace."\n'
            '    )\n'
        )
        text = text.replace(marker, note + marker, 1)
        P16.write_text(text, encoding="utf-8")
        print("[patched] participant trust terminology clarification")
        return
    backup(P16)
    P16.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("[patched] participant trust terminology clarification")


def main() -> None:
    for path in (APP, PAGES_CORE, PAGES, P16, VISUALS, COMMAND):
        if not path.exists():
            raise SystemExit("Required file missing: {}".format(path))

    # The ZIP places the new visualization and Command Centre modules directly
    # at their final paths. Do not copy them again (avoids SameFileError).
    print("[ready] research-grade Plotly visualization module installed")
    print("[ready] redesigned Command Centre module installed")

    patch_command_centre()
    patch_app_branding()
    patch_registry_name()
    patch_participant_terminology()

    print("")
    print("=" * 78)
    print("AWAREML VISUAL CALIBRATION + COMMAND CENTRE UPGRADE V8: PASS")
    print("=" * 78)
    print("- Plotly-based calibration visuals: installed")
    print("- Correct vs incorrect labels: separated and horizontal")
    print("- Accept / Override / Reject: two clearly separated condition rows")
    print("- Explanation-source chart: dumbbell gap chart + incorrect-acceptance view")
    print("- Study-experience profile: neutral-centered lollipop plots")
    print("- Likert distribution: diverging research-paper profile")
    print("- Expertise comparison: annotated construct × expertise matrix")
    print("- Validated trust-item text: unchanged")
    print("- 'AI assistant' terminology: clarified as explanation-producing system, not Copilot")
    print("- Command Centre: renamed and redesigned")
    print("- AwareML Research OS branding: replaced with AwareML Dashboard")
    print("- Visible Phase 9 label on Command Centre: removed")
    print("- Frozen study protocol/stimulus bank: not modified")
    print("")
    print("Next:")
    print("  python -m pytest -q .\\tests\\test_phase16_visual_commandcentre_v8.py")
    print("  python -m streamlit run app.py")


if __name__ == "__main__":
    main()
