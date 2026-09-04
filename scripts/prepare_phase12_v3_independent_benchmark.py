from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "journal" / "objective_selection_benchmark_v2_candidate"


def write_csv(path, fieldnames, rows=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError("Refusing to overwrite existing candidate file: {}".format(path))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "scenario_pool.NEW_UNSEEN.csv",
        ["scenario_id", "scenario", "source", "generation_batch", "notes"],
    )
    for annotator in ["A", "B", "C"]:
        write_csv(
            OUT / "annotations_{}.csv".format(annotator),
            [
                "scenario_id", "scenario", "accuracy", "runtime", "energy", "co2",
                "ambiguous_or_unclear", "annotator_id", "notes",
            ],
        )
    (OUT / "README.md").write_text(
        """# Fresh Objective-Selection V3 Benchmark Candidate\n\n"
        "Create this benchmark only after the V3 method is fixed. Do not copy Phase-12 v1 scenarios. "
        "Use new realistic statements, independent realism filtering and 2-3 independent annotators. "
        "Freeze human ground truth before any V3 inference.\n\n"
        "Recommended final size: 40-60 primary cases plus new paraphrase/adversarial material.\n"
        """,
        encoding="utf-8",
    )
    print("Created fresh V3 benchmark templates at:", OUT)
    print("No scenarios or labels were synthesized by this script.")
    print("Do not run V3 on the new cases until human ground truth is frozen.")


if __name__ == "__main__":
    main()
