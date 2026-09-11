from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path.cwd()
PAYLOAD = ROOT / "payload"
BACKUP = (
    ROOT
    / ".phase14_robustness_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def sha256(path):
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
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
    dest = BACKUP / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)


def copy_payload():
    for source in PAYLOAD.rglob("*"):
        if not source.is_file():
            continue
        dest = ROOT / source.relative_to(PAYLOAD)
        backup(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)


def patch_pages_specialist():
    path = ROOT / "awareml/ui_v2/pages_specialist.py"
    text = path.read_text(encoding="utf-8")
    original = text

    start_token = (
        '\n    section(\n'
        '        "Calibration fairness · Phase 14",'
    )
    start = text.find(start_token)

    if start != -1:
        guard = "\n    if not common_labels:"
        end = text.find(guard, start)
        if end == -1:
            raise RuntimeError(
                "Found the older inline calibration block but "
                "could not locate its end."
            )
        text = (
            text[:start]
            + "\n    render_phase14_fairness_details(results)\n"
            + text[end:]
        )

    helper = "    render_phase14_fairness_details(results)\n"
    guard = "\n    if not common_labels:"

    # Keep exactly one helper call.
    occurrences = text.count(helper)
    if occurrences == 0:
        if guard not in text:
            raise RuntimeError(
                "Could not locate Fairness Lab integration anchor."
            )
        text = text.replace(
            guard,
            "\n" + helper + guard,
            1,
        )
    elif occurrences > 1:
        first = text.find(helper)
        before = text[: first + len(helper)]
        after = text[first + len(helper):].replace(helper, "")
        text = before + after

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched: single integrated calibration fairness section"
        )


def patch_engine_runner():
    path = ROOT / "awareml/engine/runner.py"
    text = path.read_text(encoding="utf-8")
    original = text

    threshold_line = (
        "        degenerate_prediction_threshold="
        "cfg.prediction_near_constant_threshold,\n"
    )
    if threshold_line not in text:
        anchor = (
            "        calibration_bins="
            "cfg.fairness_calibration_bins,\n"
            "    )\n"
        )
        replacement = (
            "        calibration_bins="
            "cfg.fairness_calibration_bins,\n"
            "        degenerate_prediction_threshold="
            "cfg.prediction_near_constant_threshold,\n"
            "    )\n"
        )
        if anchor not in text:
            raise RuntimeError(
                "Could not locate SlidingFairness constructor."
            )
        text = text.replace(anchor, replacement, 1)

    if '"carbon_intensity_source"' not in text:
        anchor = (
            '                "measurement_failure_reason": sustainability.get(\n'
            '                    "measurement_failure_reason"\n'
            '                ),\n'
        )
        replacement = (
            anchor
            + '                "carbon_intensity_source": sustainability.get(\n'
            + '                    "carbon_intensity_source"\n'
            + '                ),\n'
        )
        if anchor not in text:
            raise RuntimeError(
                "Could not locate sustainability metadata anchor."
            )
        text = text.replace(anchor, replacement, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched: runner fairness validity + "
            "carbon-intensity provenance"
        )


def patch_positive_label_default():
    path = ROOT / "awareml/ui/pages.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "_awareml_positive_profile_key" in text:
        return

    old = (
        '            classes = df[target].dropna().unique().tolist()\n'
        '            positive_label = st.selectbox("Positive label", classes, '
        'index=1 if len(classes) > 1 else 0, key="run_positive") '
        'if classes else 1\n'
        '            _state()["positive_label"] = positive_label\n'
    )

    new = '''            classes = df[target].dropna().unique().tolist()

            # Research-safe positive-label default. The Dutch Census demo
            # profile declares occupation_binary=1 as the positive outcome.
            dataset_name = str(_state().get("dataset_name") or "")
            preferred_positive = None

            if (
                Path(dataset_name).name.lower()
                == "dutch_census_stream_awareml.csv"
                and target == "occupation_binary"
            ):
                preferred_positive = next(
                    (
                        value
                        for value in classes
                        if str(value) == "1"
                    ),
                    None,
                )

            if preferred_positive is None:
                preferred_positive = next(
                    (
                        value
                        for value in classes
                        if value == 1 or str(value) == "1"
                    ),
                    classes[-1] if classes else 1,
                )

            profile_key = "{}|{}".format(
                Path(dataset_name).name.lower(),
                target,
            )

            if (
                st.session_state.get("_awareml_positive_profile_key")
                != profile_key
            ):
                st.session_state["_awareml_positive_profile_key"] = profile_key
                st.session_state["run_positive"] = preferred_positive
            elif (
                classes
                and st.session_state.get("run_positive") not in classes
            ):
                st.session_state["run_positive"] = preferred_positive

            default_positive_index = (
                classes.index(preferred_positive)
                if classes and preferred_positive in classes
                else 0
            )

            positive_label = (
                st.selectbox(
                    "Positive label",
                    classes,
                    index=default_positive_index,
                    key="run_positive",
                )
                if classes
                else 1
            )

            _state()["positive_label"] = positive_label

            if (
                Path(dataset_name).name.lower()
                == "dutch_census_stream_awareml.csv"
                and target == "occupation_binary"
            ):
                st.caption(
                    "Dutch Census demo profile: positive label = 1."
                )
'''

    if old not in text:
        raise RuntimeError(
            "Could not locate Run Studio positive-label block."
        )

    text = text.replace(old, new, 1)

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        print(
            "Patched: deterministic/demo-profile positive label"
        )


def main():
    if not (ROOT / "app.py").exists() or not (ROOT / "awareml").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )
    if not PAYLOAD.exists():
        raise RuntimeError(
            "payload folder is missing. Extract the complete ZIP first."
        )

    before = protected_hashes()
    BACKUP.mkdir(parents=True, exist_ok=True)

    copy_payload()
    patch_pages_specialist()
    patch_engine_runner()
    patch_positive_label_default()

    after = protected_hashes()
    changed = [
        key for key in before
        if before.get(key) != after.get(key)
    ]
    if changed:
        raise RuntimeError(
            "Protected frozen/model artifacts changed unexpectedly: {}"
            .format(changed)
        )

    print("=" * 100)
    print("AwareML Phase-14 Robustness Hotfix: APPLIED")
    print("=" * 100)
    print("Backup:", BACKUP)
    print()
    print("Fixed:")
    print("  1. Zero-gap fairness now flags constant/near-constant predictions")
    print("  2. Constant probabilities are flagged beside Brier/ECE evidence")
    print("  3. Duplicate Phase-14 calibration section removed")
    print("  4. AutoStreamML GaussianNB warning flood suppressed locally")
    print("  5. Carbon intensity units corrected to gCO2/kWh")
    print("  6. CodeCarbon measurement made exclusive")
    print("  7. Dutch Census positive-label default aligned to profile (=1)")
    print("  8. Repeatability runner writes a manifest and blocks Streamlit")
    print("  9. Sustainability Lab loads matching repeated-run artifacts")
    print()
    print(
        "Protected Phase-12/13/V3/V3.1/Recommender-V2 "
        "artifacts unchanged: True"
    )
    print()
    print("Run next:")
    print(
        r"  pytest -q .\tests\test_phase14_hardening.py "
        r".\tests\test_phase14_integrated_labs.py "
        r".\tests\test_phase14_robustness_hotfix.py"
    )
    print(r"  python -m scripts.validate_phase14_hardening")
    print(r"  python -m scripts.validate_phase14_robustness_hotfix")
    print("=" * 100)


if __name__ == "__main__":
    main()
