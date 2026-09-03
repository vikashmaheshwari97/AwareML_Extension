from __future__ import annotations

from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "awareml" / "ui_v2" / "pages_core.py"
SPECIALIST = ROOT / "awareml" / "ui_v2" / "pages_specialist.py"


class PatchError(RuntimeError):
    pass


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".phase11_provenance_backup")
    if not backup.exists():
        shutil.copy2(str(path), str(backup))


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise PatchError(
            "{}: expected exactly one source pattern inside target function, found {}".format(
                label, count
            )
        )
    return text.replace(old, new, 1)


def _split_function(text: str, function_name: str, next_function_name: str):
    start_token = "def {}():".format(function_name)
    end_token = "def {}():".format(next_function_name)

    if start_token not in text:
        raise PatchError("Could not locate {}".format(start_token))
    if end_token not in text:
        raise PatchError("Could not locate {}".format(end_token))

    before, rest = text.split(start_token, 1)
    body, after = rest.split(end_token, 1)
    return before, body, after, start_token, end_token


def patch_core() -> bool:
    if not CORE.exists():
        raise PatchError("Missing {}".format(CORE))

    original = CORE.read_text(encoding="utf-8")
    before, body, after, start_token, end_token = _split_function(
        original,
        "decision_space_page",
        "run_studio_v2_page",
    )

    provenance_anchor = "    if not dataset_ready():\n"
    provenance_block = """    st.markdown(
        \"\"\"
        <div class=\"r9-callout\">
          <b>PRE-RUN PREFERENCE RECOMMENDATION</b><br>
          <b>Evidence source:</b> predicted framework outcomes from frozen ML Recommender V2.<br>
          <b>Preference source:</b> the manual Accuracy / Runtime / Energy / CO₂ sliders on this page.<br>
          <b>Ranking engine:</b> ML Recommender V2 preference-aware reranking.<br>
          <b>Framework execution:</b> not required; these are predictions, not observed results.
        </div>
        \"\"\",
        unsafe_allow_html=True,
    )

    if not dataset_ready():
"""
    if "PRE-RUN PREFERENCE RECOMMENDATION" not in body:
        body = _replace_once(
            body,
            provenance_anchor,
            provenance_block,
            "3d-provenance",
        )

    replacements = [
        (
            '        st.metric("Recommended", str(top["framework"]), "Rank #1")',
            '        st.metric("Pre-run recommended framework", str(top["framework"]), "Predicted rank #1")',
            "recommended-card",
        ),
        (
            '        st.metric("Utility", fmt(top["utility"], 4))',
            '        st.metric("Pre-run preference utility", fmt(top["utility"], 4))',
            "utility-card",
        ),
        (
            '        st.metric("Selected accuracy", fmt(selected_row["accuracy"], 4))',
            '        st.metric("{} predicted accuracy".format(selected), fmt(selected_row["accuracy"], 4))',
            "accuracy-card",
        ),
        (
            '        st.metric("Selected runtime", fmt(selected_row["runtime"], 3, " s"))',
            '        st.metric("{} predicted runtime".format(selected), fmt(selected_row["runtime"], 3, " s"))',
            "runtime-card",
        ),
        (
            '        st.metric("Selected energy", fmt(selected_row["energy"], 6, " kWh"))',
            '        st.metric("{} predicted energy".format(selected), fmt(selected_row["energy"], 6, " kWh"))',
            "energy-card",
        ),
        (
            '        st.metric("Selected CO₂", fmt(selected_row.get("co2"), 6, " kg"))',
            '        st.metric("{} predicted CO₂".format(selected), fmt(selected_row.get("co2"), 6, " kg"))',
            "co2-card",
        ),
    ]

    for old, new, label in replacements:
        body = _replace_once(body, old, new, label)

    caption_anchor = "    view_mode = st.segmented_control(\n"
    caption_block = """    st.caption(
        \"Recommended framework and inspected framework are different concepts: \"
        \"the first two cards describe the current predicted rank #1, while the \"
        \"four objective cards describe the framework chosen in 'Inspect framework'.\"
    )

    view_mode = st.segmented_control(
"""
    if "Recommended framework and inspected framework are different concepts" not in body:
        body = _replace_once(
            body,
            caption_anchor,
            caption_block,
            "recommendation-inspection-caption",
        )

    updated = before + start_token + body + end_token + after
    if updated != original:
        _backup(CORE)
        CORE.write_text(updated, encoding="utf-8")
        return True
    return False


def patch_specialist() -> bool:
    if not SPECIALIST.exists():
        raise PatchError("Missing {}".format(SPECIALIST))

    original = SPECIALIST.read_text(encoding="utf-8")

    # Scope the patch to decision_lab_v2_page only. Use the next top-level def as boundary.
    start_token = "def decision_lab_v2_page():"
    if start_token not in original:
        raise PatchError("Could not locate {}".format(start_token))

    before, rest = original.split(start_token, 1)

    # Find the next top-level function definition without depending on its name.
    next_index = rest.find("\ndef ")
    if next_index == -1:
        body = rest
        after = ""
    else:
        body = rest[: next_index + 1]
        after = rest[next_index + 1 :]

    section_anchor = '    section("Observed-objective weights", "Set the importance of each measured criterion. Weights are normalized by the ranking engine over available evidence.")\n'
    provenance_block = """    st.markdown(
        \"\"\"
        <div class=\"r9-callout\">
          <b>OBSERVED POST-RUN RECOMMENDATION</b><br>
          <b>Evidence source:</b> outcomes actually measured in the current five-framework benchmark.<br>
          <b>Preference source:</b> the six post-run sliders on this page.<br>
          <b>Ranking engine:</b> observed multi-objective utility + near-Pareto analysis.<br>
          <b>Framework execution:</b> completed; this page ranks measured evidence rather than pre-run predictions.
        </div>
        \"\"\",
        unsafe_allow_html=True,
    )

    section("Observed-objective weights", "Set the importance of each measured criterion. Weights are normalized by the ranking engine over available evidence.")
"""
    if "OBSERVED POST-RUN RECOMMENDATION" not in body:
        body = _replace_once(
            body,
            section_anchor,
            provenance_block,
            "decision-provenance",
        )

    body = _replace_once(
        body,
        '    cards[0].metric("Recommended framework", top.framework)',
        '    cards[0].metric("Post-run recommended framework", top.framework)',
        "decision-recommended-card",
    )

    updated = before + start_token + body + after
    if updated != original:
        _backup(SPECIALIST)
        SPECIALIST.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed_core = patch_core()
    changed_specialist = patch_specialist()

    print("=" * 72)
    print("AwareML Phase 11 recommendation provenance clarity v2: APPLIED")
    print("=" * 72)
    print("pages_core.py changed:", changed_core)
    print("pages_specialist.py changed:", changed_specialist)
    print()
    print("3D Decision Space  -> PRE-RUN PREFERENCE RECOMMENDATION")
    print("Copilot Workspace  -> SCENARIO-CONDITIONED PRE-RUN RECOMMENDATION")
    print("Decision Lab       -> OBSERVED POST-RUN RECOMMENDATION")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except PatchError as exc:
        print("PATCH FAILED:", exc, file=sys.stderr)
        raise SystemExit(1)
