from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v1" / "human"
OUT = ROOT / "data" / "journal" / "objective_selection_v31_development" / "provenance"


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row["scenario_id"]: row for row in csv.DictReader(fh)}


def norm_note(value):
    return " ".join(str(value or "").strip().lower().split())


def main():
    a = read(BASE / "annotations_A.csv")
    b = read(BASE / "annotations_B.csv")
    c = read(BASE / "annotations_C.csv")
    ids = sorted(set(a) & set(b) & set(c))

    ab = ac = bc = all3 = 0
    examples = []
    for sid in ids:
        na, nb, nc = map(norm_note, [a[sid].get("notes"), b[sid].get("notes"), c[sid].get("notes")])
        ab += int(bool(na) and na == nb)
        ac += int(bool(na) and na == nc)
        bc += int(bool(nb) and nb == nc)
        all3 += int(bool(na) and na == nb == nc)
        if bool(na) and na == nb == nc and len(examples) < 8:
            examples.append({"scenario_id": sid, "note": a[sid].get("notes")})

    payload = {
        "artifact": "phase12_annotation_provenance_audit",
        "rows": len(ids),
        "exact_nonempty_note_matches": {
            "A_equals_B": ab,
            "A_equals_C": ac,
            "B_equals_C": bc,
            "A_equals_B_equals_C": all3,
        },
        "examples": examples,
        "interpretation": (
            "High note-text overlap is a provenance question, not proof of non-independent labels. "
            "Before publication, document whether labels were independently collected and whether notes were normalized later."
        ),
        "researcher_confirmation_required": True,
        "do_not_rewrite_raw_annotation_files": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "annotation_provenance_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "ANNOTATION_PROVENANCE_REVIEW.md").write_text(
        "# Phase-12 annotation provenance review\n\n"
        "This audit does **not** infer how the annotations were produced. It records a reproducible observation: "
        "many free-text notes are identical across annotator files.\n\n"
        "Before journal submission, confirm one of the following with supporting records:\n\n"
        "- labels and notes were independently collected; or\n"
        "- objective labels were independently collected and notes were subsequently normalized for readability.\n\n"
        "Do not silently edit the frozen/raw annotation files. Preserve original evidence.\n",
        encoding="utf-8",
    )

    print("=" * 76)
    print("Phase-12 annotation provenance audit: COMPLETE")
    print("=" * 76)
    print("Rows:", len(ids))
    print("A == B exact non-empty notes:", ab)
    print("A == C exact non-empty notes:", ac)
    print("B == C exact non-empty notes:", bc)
    print("All three exact non-empty notes:", all3)
    print("Researcher confirmation required: True")
    print("Frozen annotation files modified: False")
    print("=" * 76)


if __name__ == "__main__":
    main()
