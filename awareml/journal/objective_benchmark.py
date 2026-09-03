from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


OBJECTIVES = ("Accuracy", "Runtime", "Energy", "CO2")
OBJECTIVE_COLUMNS = ("accuracy", "runtime", "energy", "co2")
EXPLICIT_OBJECTIVE_RE = re.compile(r"\b(?:accuracy|runtime|energy|co2|co₂)\b", re.I)


class Phase12Error(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_bool(value: Any, *, allow_blank: bool = False) -> Optional[bool]:
    text = "" if value is None else str(value).strip().lower()
    if allow_blank and text == "":
        return None
    if text in {"1", "true", "yes", "y", "keep", "include", "checked"}:
        return True
    if text in {"0", "false", "no", "n", "drop", "exclude", "unchecked"}:
        return False
    raise Phase12Error("Expected yes/no or 1/0, got {!r}".format(value))


def objective_set_from_columns(row: Mapping[str, Any]) -> Set[str]:
    selected: Set[str] = set()
    for objective, column in zip(OBJECTIVES, OBJECTIVE_COLUMNS):
        if parse_bool(row.get(column), allow_blank=False):
            selected.add(objective)
    return selected


def labels_to_columns(labels: Iterable[str]) -> Dict[str, int]:
    label_set = set(labels)
    return {
        column: int(objective in label_set)
        for objective, column in zip(OBJECTIVES, OBJECTIVE_COLUMNS)
    }


def validate_generated_design(root: Path) -> Dict[str, Any]:
    root = Path(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1" / "design"
    candidates = read_csv(base / "candidate_pool.generated.csv")
    intents = read_csv(base / "candidate_generation_intent.PRIVATE.csv")
    paraphrases = read_csv(base / "paraphrase_families.generated.csv")
    adversarial = read_csv(base / "adversarial_set.csv")
    provenance = read_json(base / "generation_provenance.json")

    if len(candidates) != 120:
        raise Phase12Error("Expected 120 generated candidates, found {}.".format(len(candidates)))

    ids = [row["scenario_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise Phase12Error("Generated candidate IDs are not unique.")

    scenario_texts = [row["scenario"].strip() for row in candidates]
    if len(scenario_texts) != len(set(scenario_texts)):
        raise Phase12Error("Generated candidate scenarios are not unique.")

    for row in candidates:
        match = EXPLICIT_OBJECTIVE_RE.search(row["scenario"])
        if match:
            raise Phase12Error(
                "Candidate {} explicitly names frozen objective {!r}.".format(
                    row["scenario_id"], match.group(0)
                )
            )

    intent_map = {row["scenario_id"]: row for row in intents}
    if set(intent_map) != set(ids):
        raise Phase12Error("Candidate intent map does not match public candidate IDs.")

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for row in intents:
        k = int(row["intended_k_prime"])
        if k not in counts:
            raise Phase12Error("Invalid intended k-prime: {}".format(k))
        labels = [x for x in row["generation_intent"].split("|") if x]
        if len(labels) != k or any(label not in OBJECTIVES for label in labels):
            raise Phase12Error("Invalid private generation intent for {}.".format(row["scenario_id"]))
        counts[k] += 1
    if counts != {1: 30, 2: 30, 3: 30, 4: 30}:
        raise Phase12Error("k-prime candidate design is not balanced: {}".format(counts))

    family_ids = sorted({row["family_id"] for row in paraphrases})
    if len(family_ids) != 10:
        raise Phase12Error("Expected 10 paraphrase families.")
    for family_id in family_ids:
        rows = [r for r in paraphrases if r["family_id"] == family_id]
        base_count = sum(1 for r in rows if r["variant_type"] == "base")
        para_count = sum(1 for r in rows if r["variant_type"] == "paraphrase")
        if base_count != 1 or para_count != 5:
            raise Phase12Error(
                "{} must contain 1 base + 5 paraphrases.".format(family_id)
            )

    if not 10 <= len(adversarial) <= 15:
        raise Phase12Error("Adversarial set must contain 10-15 cases.")

    if provenance.get("evaluated_model_used_for_generation") is not False:
        raise Phase12Error("Evaluated journal LLM must not generate Phase-12 candidates.")
    if provenance.get("generator_model") != "GPT-5.6 Sol":
        raise Phase12Error("Unexpected external generation model provenance.")

    return {
        "candidate_count": len(candidates),
        "k_prime_design_counts": {str(k): counts[k] for k in counts},
        "paraphrase_families": len(family_ids),
        "paraphrase_evaluations": sum(
            1 for r in paraphrases if r["variant_type"] == "paraphrase"
        ),
        "adversarial_cases": len(adversarial),
        "generator_model": provenance.get("generator_model"),
    }


def phase10_protocol_info(root: Path) -> Dict[str, Any]:
    root = Path(root)
    marker = root / "data" / "journal" / "active_protocol.txt"
    if not marker.exists():
        raise Phase12Error("Phase-10 active protocol marker is missing.")
    rel = marker.read_text(encoding="utf-8").strip()
    path = root / "data" / "journal" / rel
    if not path.exists():
        raise Phase12Error("Phase-10 frozen protocol is missing.")
    payload = read_json(path)
    if payload.get("release_status") != "frozen":
        raise Phase12Error("Phase-10 protocol is not frozen.")
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "journal_llm": payload.get("journal_llm"),
        "heldout_policy": payload.get("heldout_policy"),
    }


def phase11_selector_info(root: Path) -> Dict[str, Any]:
    root = Path(root)
    marker = root / "data" / "journal" / "active_objective_selector.txt"
    if not marker.exists():
        raise Phase12Error("Phase-11 active objective selector marker is missing.")

    rel = marker.read_text(encoding="utf-8").strip()
    candidate_paths = [
        root / "data" / "journal" / rel,
        root / rel,
    ]
    manifest = None
    for path in candidate_paths:
        if path.exists():
            manifest = path
            break

    if manifest is None:
        fallback = root / "data" / "journal" / "objective_selection_v2" / "manifest.json"
        if fallback.exists():
            manifest = fallback

    if manifest is None:
        raise Phase12Error("Could not locate frozen Phase-11 selector manifest.")

    payload = read_json(manifest)
    if payload.get("release_status") not in {None, "frozen"}:
        raise Phase12Error("Phase-11 selector manifest is not frozen.")

    return {
        "path": str(manifest.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(manifest),
        "payload": payload,
    }


def freeze_design(root: Path) -> Dict[str, Any]:
    root = Path(root)
    design_summary = validate_generated_design(root)
    phase10 = phase10_protocol_info(root)
    phase11 = phase11_selector_info(root)

    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    design_dir = base / "design"
    output_dir = base / "frozen_design"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = output_dir / "design_manifest.json"
    existing_sha = output_dir / "design_manifest.json.sha256"
    if existing_manifest.exists() and existing_sha.exists():
        payload = validate_design_freeze(root)
        return {
            "manifest": existing_manifest,
            "sha256": sha256_file(existing_manifest),
            "summary": payload["design_summary"],
        }

    premature_outputs = [
        base / "results" / "llm_outputs.jsonl",
        base / "results" / "paraphrase_llm_outputs.jsonl",
        base / "results" / "adversarial_llm_outputs.jsonl",
    ]
    present = [path for path in premature_outputs if path.exists()]
    if present:
        raise Phase12Error(
            "Phase-12 design must be frozen before evaluated-LLaMA outputs exist: {}".format(
                ", ".join(path.name for path in present)
            )
        )

    protected = [
        design_dir / "candidate_pool.generated.csv",
        design_dir / "candidate_generation_intent.PRIVATE.csv",
        design_dir / "paraphrase_families.generated.csv",
        design_dir / "paraphrase_generation_intent.PRIVATE.csv",
        design_dir / "adversarial_set.csv",
        design_dir / "generation_provenance.json",
        root / "prompts" / "phase12_external_scenario_generation_v1.txt",
    ]

    hashes = {}
    for path in protected:
        if not path.exists():
            raise Phase12Error("Missing Phase-12 design asset: {}".format(path))
        hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)

    manifest = {
        "artifact": "objective_selection_benchmark_v1_design",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "phase10_protocol": phase10,
        "phase11_selector": {
            "path": phase11["path"],
            "sha256": phase11["sha256"],
        },
        "design_summary": design_summary,
        "file_sha256": hashes,
        "ground_truth_source": "independent_human_annotation_only",
        "generation_intent_is_ground_truth": False,
        "evaluated_llm_outputs_created_before_design_freeze": False,
        "heldout_dataset_contents_used": False,
    }
    manifest_path = output_dir / "design_manifest.json"
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    (output_dir / "design_manifest.json.sha256").write_text(
        "{}  design_manifest.json\n".format(digest), encoding="utf-8"
    )
    marker = base / "active_design.txt"
    marker.write_text(
        "objective_selection_benchmark_v1/frozen_design/design_manifest.json\n",
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "sha256": digest, "summary": design_summary}


def validate_design_freeze(root: Path) -> Dict[str, Any]:
    root = Path(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    marker = base / "active_design.txt"
    if not marker.exists():
        raise Phase12Error("Phase-12 design has not been frozen.")
    rel = marker.read_text(encoding="utf-8").strip()
    path = root / "data" / "journal" / rel
    if not path.exists():
        # Marker may already be relative to data/journal.
        path = root / "data" / "journal" / rel
    if not path.exists():
        fallback = base / "frozen_design" / "design_manifest.json"
        if fallback.exists():
            path = fallback
    manifest = read_json(path)
    sha_path = Path(str(path) + ".sha256")
    if not sha_path.exists():
        raise Phase12Error("Phase-12 design checksum is missing.")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise Phase12Error("Phase-12 design manifest checksum mismatch.")

    for rel_path, expected_sha in manifest["file_sha256"].items():
        asset = root / rel_path
        if not asset.exists() or sha256_file(asset) != expected_sha:
            raise Phase12Error("Frozen design asset changed: {}".format(rel_path))
    return manifest


def prepare_annotation_packets(root: Path, *, force: bool = False) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    human = base / "human"
    filter_rows = read_csv(human / "realism_filter.csv")
    candidate_map = {
        row["scenario_id"]: row
        for row in read_csv(base / "design" / "candidate_pool.generated.csv")
    }

    accepted_generated: List[Dict[str, str]] = []
    incomplete_filter = []
    for row in filter_rows:
        if str(row.get("keep", "")).strip() == "":
            incomplete_filter.append(row["scenario_id"])
            continue
        keep = parse_bool(row["keep"])
        ambiguity = parse_bool(row.get("ambiguous_or_unclear"), allow_blank=True)
        if keep and ambiguity is True:
            raise Phase12Error(
                "{} is marked both keep=yes and ambiguous=yes.".format(row["scenario_id"])
            )
        if keep:
            if not str(row.get("realism_1_to_5", "")).strip():
                raise Phase12Error(
                    "{} is retained but realism score is blank.".format(row["scenario_id"])
                )
            score = int(float(row["realism_1_to_5"]))
            if score < 1 or score > 5:
                raise Phase12Error("Realism score must be 1-5.")
            if not str(row.get("filterer_id", "")).strip():
                raise Phase12Error("Retained scenario {} lacks filterer_id.".format(row["scenario_id"]))
            accepted_generated.append(candidate_map[row["scenario_id"]])

    if incomplete_filter:
        raise Phase12Error(
            "Human realism filtering is incomplete for {} candidates; first missing: {}.".format(
                len(incomplete_filter), ", ".join(incomplete_filter[:10])
            )
        )
    if not 40 <= len(accepted_generated) <= 50:
        raise Phase12Error(
            "Human realism filter must retain 40-50 generated scenarios; retained {}.".format(
                len(accepted_generated)
            )
        )

    human_written_rows = read_csv(human / "human_written_scenarios.csv")
    accepted_human: List[Dict[str, str]] = []
    for row in human_written_rows:
        scenario = str(row.get("scenario", "")).strip()
        if not scenario:
            continue
        if EXPLICIT_OBJECTIVE_RE.search(scenario):
            raise Phase12Error(
                "Human-written scenario {} explicitly names a frozen objective.".format(
                    row["human_scenario_id"]
                )
            )
        if not parse_bool(
            row.get("confirm_not_explicitly_naming_frozen_objectives"),
            allow_blank=False,
        ):
            raise Phase12Error(
                "Human-written scenario {} has not been confirmed.".format(
                    row["human_scenario_id"]
                )
            )
        if not str(row.get("author_id", "")).strip():
            raise Phase12Error(
                "Human-written scenario {} lacks author_id.".format(
                    row["human_scenario_id"]
                )
            )
        accepted_human.append(
            {
                "scenario_id": row["human_scenario_id"],
                "scenario": scenario,
                "domain": "human_written",
                "phrasing_style": "human_written",
                "source": "genuine_human_written",
            }
        )

    if len(accepted_human) > 15:
        raise Phase12Error("At most 15 human-written scenarios are supported.")

    final_rows = accepted_generated + accepted_human
    if not 50 <= len(final_rows) <= 65:
        raise Phase12Error(
            "Final annotation pool must contain 50-65 scenarios; currently {} "
            "({} generated + {} human-written).".format(
                len(final_rows), len(accepted_generated), len(accepted_human)
            )
        )

    packet_fields = [
        "scenario_id",
        "scenario",
        "accuracy",
        "runtime",
        "energy",
        "co2",
        "ambiguous",
        "notes",
        "annotator_id",
    ]
    for annotator in ("A", "B", "C"):
        path = human / "annotations_{}.csv".format(annotator)
        if path.exists() and not force:
            raise Phase12Error(
                "{} already exists. Use --force only before annotators start.".format(path.name)
            )
        rows = []
        for row in final_rows:
            rows.append(
                {
                    "scenario_id": row["scenario_id"],
                    "scenario": row["scenario"],
                    "accuracy": "",
                    "runtime": "",
                    "energy": "",
                    "co2": "",
                    "ambiguous": "",
                    "notes": "",
                    "annotator_id": "",
                }
            )
        write_csv(path, rows, packet_fields)

    write_csv(
        human / "final_annotation_pool.csv",
        final_rows,
        ["scenario_id", "scenario", "domain", "phrasing_style", "source"],
    )

    return {
        "generated_retained": len(accepted_generated),
        "human_written": len(accepted_human),
        "final_annotation_pool": len(final_rows),
        "annotation_files": [
            str((human / "annotations_{}.csv".format(x)).relative_to(root)).replace("\\", "/")
            for x in ("A", "B", "C")
        ],
    }


def _load_completed_annotator(path: Path, expected_ids: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    rows = read_csv(path)
    if [r["scenario_id"] for r in rows] != list(expected_ids):
        raise Phase12Error("{} scenario order/IDs differ from the frozen packet.".format(path.name))

    annotator_ids = set()
    parsed_rows = {}
    for row in rows:
        fields = [row.get(c, "") for c in OBJECTIVE_COLUMNS] + [row.get("ambiguous", "")]
        if any(str(v).strip() == "" for v in fields):
            return None
        annotator_id = str(row.get("annotator_id", "")).strip()
        if not annotator_id:
            return None
        annotator_ids.add(annotator_id)
        parsed_rows[row["scenario_id"]] = {
            "labels": objective_set_from_columns(row),
            "ambiguous": bool(parse_bool(row["ambiguous"])),
            "notes": row.get("notes", ""),
        }
    if len(annotator_ids) != 1:
        raise Phase12Error("{} must contain one consistent annotator_id.".format(path.name))
    return {
        "file": path.name,
        "annotator_id": next(iter(annotator_ids)),
        "rows": parsed_rows,
    }


def fleiss_kappa_binary(ratings: Sequence[Sequence[int]]) -> Optional[float]:
    if not ratings:
        return None
    m = len(ratings[0])
    if m < 2:
        return None
    n = len(ratings)
    yes_total = sum(sum(row) for row in ratings)
    p_yes = yes_total / float(n * m)
    p_no = 1.0 - p_yes
    p_e = p_yes * p_yes + p_no * p_no

    p_i = []
    for row in ratings:
        yes = sum(row)
        no = m - yes
        agreement = (yes * yes + no * no - m) / float(m * (m - 1))
        p_i.append(agreement)
    p_bar = sum(p_i) / float(n)

    denom = 1.0 - p_e
    if abs(denom) < 1e-12:
        return None
    return (p_bar - p_e) / denom


def _jaccard(a: Set[str], b: Set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / float(len(union))


def aggregate_annotations(root: Path) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    human = base / "human"
    pool = read_csv(human / "final_annotation_pool.csv")
    expected_ids = [r["scenario_id"] for r in pool]
    scenario_map = {r["scenario_id"]: r for r in pool}

    completed = []
    for name in ("A", "B", "C"):
        parsed = _load_completed_annotator(
            human / "annotations_{}.csv".format(name),
            expected_ids,
        )
        if parsed is not None:
            completed.append(parsed)

    if len(completed) < 2:
        raise Phase12Error(
            "At least two complete independent annotation files are required; found {}.".format(
                len(completed)
            )
        )
    annotator_ids = [a["annotator_id"] for a in completed]
    if len(set(annotator_ids)) != len(annotator_ids):
        raise Phase12Error("Annotator IDs must be distinct across completed files.")

    m = len(completed)
    kappas: Dict[str, Optional[float]] = {}
    for objective in OBJECTIVES:
        ratings = []
        for sid in expected_ids:
            ratings.append(
                [
                    int(objective in ann["rows"][sid]["labels"])
                    for ann in completed
                ]
            )
        kappas[objective] = fleiss_kappa_binary(ratings)

    unanimous = 0
    pairwise_jaccard = []
    for sid in expected_ids:
        label_sets = [ann["rows"][sid]["labels"] for ann in completed]
        if all(labels == label_sets[0] for labels in label_sets[1:]):
            unanimous += 1
        for i, j in itertools.combinations(range(m), 2):
            pairwise_jaccard.append(_jaccard(label_sets[i], label_sets[j]))

    resolved = []
    hard = []
    for sid in expected_ids:
        scenario = scenario_map[sid]["scenario"]
        ambiguity_votes = sum(
            int(ann["rows"][sid]["ambiguous"]) for ann in completed
        )
        if ambiguity_votes > m / 2.0:
            hard.append(
                {
                    "scenario_id": sid,
                    "scenario": scenario,
                    "reason": "majority_marked_ambiguous",
                    "annotator_count": m,
                }
            )
            continue

        selected = []
        ties = []
        vote_detail = {}
        for objective in OBJECTIVES:
            votes = sum(
                int(objective in ann["rows"][sid]["labels"]) for ann in completed
            )
            vote_detail[objective] = votes
            if votes > m / 2.0:
                selected.append(objective)
            elif m % 2 == 0 and votes == m / 2:
                ties.append(objective)

        if ties:
            hard.append(
                {
                    "scenario_id": sid,
                    "scenario": scenario,
                    "reason": "objective_vote_tie",
                    "tied_objectives": ties,
                    "votes": vote_detail,
                    "annotator_count": m,
                }
            )
            continue

        if not selected:
            hard.append(
                {
                    "scenario_id": sid,
                    "scenario": scenario,
                    "reason": "no_objective_reached_majority",
                    "votes": vote_detail,
                    "annotator_count": m,
                }
            )
            continue

        resolved.append(
            {
                "scenario_id": sid,
                "scenario": scenario,
                "source": scenario_map[sid]["source"],
                "ground_truth_objectives": selected,
                "k_prime": len(selected),
                "annotator_count": m,
                "votes": vote_detail,
            }
        )

    agreement = {
        "annotator_count": m,
        "annotator_ids": annotator_ids,
        "scenario_count": len(expected_ids),
        "per_objective_fleiss_kappa": kappas,
        "mean_defined_fleiss_kappa": (
            sum(v for v in kappas.values() if v is not None)
            / max(1, sum(1 for v in kappas.values() if v is not None))
        ),
        "unanimous_full_set_agreement_rate": unanimous / float(len(expected_ids)),
        "mean_pairwise_jaccard": (
            sum(pairwise_jaccard) / float(len(pairwise_jaccard))
            if pairwise_jaccard
            else None
        ),
        "resolved_ground_truth": len(resolved),
        "hard_cases": len(hard),
        "hard_case_policy": "excluded_from_primary_metrics_and_reported_separately",
    }

    results = base / "results"
    write_jsonl(results / "ground_truth.jsonl", resolved)
    write_jsonl(results / "hard_cases.annotation.jsonl", hard)
    write_json(results / "annotation_agreement.json", agreement)

    return {
        "agreement": agreement,
        "ground_truth": resolved,
        "hard_cases": hard,
    }


def validate_ground_truth(root: Path) -> Dict[str, Any]:
    root = Path(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1" / "results"
    gt_path = base / "ground_truth.jsonl"
    agreement_path = base / "annotation_agreement.json"
    if not gt_path.exists() or not agreement_path.exists():
        raise Phase12Error("Human ground truth/agreement artifacts are not ready.")
    gt = read_jsonl(gt_path)
    agreement = read_json(agreement_path)
    if not 50 <= len(gt) <= 65:
        raise Phase12Error(
            "Resolved primary benchmark must contain 50-65 cases; currently {}. "
            "Resolve/remove hard cases or add human scenarios before LLaMA evaluation.".format(
                len(gt)
            )
        )
    for row in gt:
        labels = row["ground_truth_objectives"]
        if not labels or any(label not in OBJECTIVES for label in labels):
            raise Phase12Error("Invalid ground-truth labels for {}.".format(row["scenario_id"]))
        if int(row["k_prime"]) != len(labels):
            raise Phase12Error("Invalid k-prime for {}.".format(row["scenario_id"]))
    return {"count": len(gt), "agreement": agreement, "rows": gt}


def validate_paraphrase_review(root: Path) -> Dict[str, Any]:
    root = Path(root)
    path = (
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "human"
        / "paraphrase_family_review.csv"
    )
    rows = read_csv(path)
    if len(rows) != 10:
        raise Phase12Error("Expected 10 paraphrase family review rows.")
    reviewers = set()
    targets = {}
    for row in rows:
        if any(str(row.get(c, "")).strip() == "" for c in OBJECTIVE_COLUMNS):
            raise Phase12Error("Paraphrase family {} lacks human objective labels.".format(row["family_id"]))
        if not parse_bool(row.get("all_5_paraphrases_preserve_meaning"), allow_blank=False):
            raise Phase12Error(
                "All five paraphrases must be human-confirmed as meaning-preserving; {} failed review.".format(
                    row["family_id"]
                )
            )
        reviewer = str(row.get("reviewer_id", "")).strip()
        if not reviewer:
            raise Phase12Error("Paraphrase family {} lacks reviewer_id.".format(row["family_id"]))
        reviewers.add(reviewer)
        labels = sorted(objective_set_from_columns(row), key=OBJECTIVES.index)
        if not labels:
            raise Phase12Error("Paraphrase family {} has no target objectives.".format(row["family_id"]))
        targets[row["family_id"]] = labels
    return {"targets": targets, "reviewer_ids": sorted(reviewers)}


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_predictions(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pred_map = {row["scenario_id"]: row for row in predictions}
    if len(pred_map) != len(predictions):
        raise Phase12Error("Prediction scenario IDs are not unique.")
    if set(pred_map) != {row["scenario_id"] for row in ground_truth}:
        raise Phase12Error("Prediction IDs do not exactly match ground truth.")

    per_objective = {}
    micro_true: List[int] = []
    micro_pred: List[int] = []
    exact = 0
    by_k: Dict[int, Dict[str, Any]] = {}

    for objective in OBJECTIVES:
        y_true = []
        y_pred = []
        for gt in ground_truth:
            sid = gt["scenario_id"]
            true_set = set(gt["ground_truth_objectives"])
            pred_set = set(pred_map[sid].get("selected_objectives") or [])
            y_true.append(int(objective in true_set))
            y_pred.append(int(objective in pred_set))
        per_objective[objective] = binary_metrics(y_true, y_pred)
        micro_true.extend(y_true)
        micro_pred.extend(y_pred)

    for gt in ground_truth:
        sid = gt["scenario_id"]
        true_set = set(gt["ground_truth_objectives"])
        pred_set = set(pred_map[sid].get("selected_objectives") or [])
        is_exact = true_set == pred_set
        exact += int(is_exact)
        k = int(gt["k_prime"])
        bucket = by_k.setdefault(
            k,
            {"n": 0, "exact": 0, "jaccard_sum": 0.0},
        )
        bucket["n"] += 1
        bucket["exact"] += int(is_exact)
        bucket["jaccard_sum"] += _jaccard(true_set, pred_set)

    micro = binary_metrics(micro_true, micro_pred)
    macro_f1 = sum(per_objective[o]["f1"] for o in OBJECTIVES) / float(len(OBJECTIVES))

    by_k_report = {}
    for k in sorted(by_k):
        b = by_k[k]
        by_k_report[str(k)] = {
            "n": b["n"],
            "exact_match_rate": b["exact"] / float(b["n"]),
            "mean_jaccard": b["jaccard_sum"] / float(b["n"]),
        }

    statuses = {}
    for row in predictions:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

    return {
        "n": len(ground_truth),
        "per_objective": per_objective,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "macro_f1": macro_f1,
        "exact_match_rate": exact / float(len(ground_truth)),
        "by_k_prime": by_k_report,
        "prediction_status_counts": statuses,
    }


def classify_adversarial_case(
    expected_status: str,
    expected_behavior: str,
    actual_status: str,
    selected_objectives: Sequence[str],
) -> str:
    selected = list(selected_objectives or [])
    if actual_status == "malformed":
        return "silent_failure"
    if expected_status == "contradictory":
        return (
            "contradiction_handling"
            if actual_status == "contradictory"
            else ("over_selection" if selected else "contradiction_missed")
        )
    if expected_status == "out_of_scope":
        return (
            "out_of_scope_handling"
            if actual_status == "out_of_scope"
            else ("over_selection" if selected else "out_of_scope_missed")
        )
    if expected_status == "ambiguous":
        if actual_status == "ambiguous" and not selected:
            return "sensible_default"
        if selected:
            return "over_selection"
        return "ambiguous_handling_failure"
    return expected_behavior



def freeze_ground_truth(root: Path) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    gt = validate_ground_truth(root)
    paraphrase_review = validate_paraphrase_review(root)

    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    output_dir = base / "frozen_ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = output_dir / "ground_truth_manifest.json"
    existing_sha = output_dir / "ground_truth_manifest.json.sha256"
    if existing_manifest.exists() and existing_sha.exists():
        payload = validate_ground_truth_freeze(root)
        return {
            "manifest": existing_manifest,
            "sha256": sha256_file(existing_manifest),
            "payload": payload,
        }

    premature_outputs = [
        base / "results" / "llm_outputs.jsonl",
        base / "results" / "paraphrase_llm_outputs.jsonl",
        base / "results" / "adversarial_llm_outputs.jsonl",
    ]
    present = [path for path in premature_outputs if path.exists()]
    if present:
        raise Phase12Error(
            "Human ground truth must be frozen before evaluated-LLaMA outputs exist: {}".format(
                ", ".join(path.name for path in present)
            )
        )

    human_dir = base / "human"
    results_dir = base / "results"
    protected = [
        human_dir / "realism_filter.csv",
        human_dir / "human_written_scenarios.csv",
        human_dir / "final_annotation_pool.csv",
        human_dir / "paraphrase_family_review.csv",
        results_dir / "ground_truth.jsonl",
        results_dir / "hard_cases.annotation.jsonl",
        results_dir / "annotation_agreement.json",
    ]

    # Preserve only completed independent annotator files.
    pool = read_csv(human_dir / "final_annotation_pool.csv")
    ids = [row["scenario_id"] for row in pool]
    completed_annotators = []
    for name in ("A", "B", "C"):
        path = human_dir / "annotations_{}.csv".format(name)
        parsed = _load_completed_annotator(path, ids) if path.exists() else None
        if parsed is not None:
            protected.append(path)
            completed_annotators.append(
                {"file": path.name, "annotator_id": parsed["annotator_id"]}
            )

    if len(completed_annotators) < 2:
        raise Phase12Error("At least two completed annotator files are required.")

    hashes = {}
    for path in protected:
        if not path.exists():
            raise Phase12Error("Ground-truth freeze input missing: {}".format(path))
        hashes[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)

    payload = {
        "artifact": "objective_selection_benchmark_v1_ground_truth",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "resolved_primary_cases": gt["count"],
        "annotation_agreement": gt["agreement"],
        "completed_annotators": completed_annotators,
        "paraphrase_reviewer_ids": paraphrase_review["reviewer_ids"],
        "ground_truth_source": "independent_human_annotation",
        "generation_intent_used_as_ground_truth": False,
        "llama_outputs_existed_before_ground_truth_freeze": False,
        "heldout_dataset_contents_used": False,
        "file_sha256": hashes,
        "design_manifest_sha256": sha256_file(
            base / "frozen_design" / "design_manifest.json"
        ),
    }

    manifest_path = output_dir / "ground_truth_manifest.json"
    write_json(manifest_path, payload)
    digest = sha256_file(manifest_path)
    (output_dir / "ground_truth_manifest.json.sha256").write_text(
        "{}  ground_truth_manifest.json\n".format(digest),
        encoding="utf-8",
    )
    (base / "active_ground_truth.txt").write_text(
        "objective_selection_benchmark_v1/frozen_ground_truth/ground_truth_manifest.json\n",
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "sha256": digest, "payload": payload}


def validate_ground_truth_freeze(root: Path) -> Dict[str, Any]:
    root = Path(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    marker = base / "active_ground_truth.txt"
    if not marker.exists():
        raise Phase12Error(
            "Human ground truth has not been frozen. Run freeze_phase12_ground_truth.py "
            "before any evaluated-LLaMA benchmark."
        )
    rel = marker.read_text(encoding="utf-8").strip()
    manifest_path = root / "data" / "journal" / rel
    if not manifest_path.exists():
        raise Phase12Error("Frozen human ground-truth manifest is missing.")
    sha_path = Path(str(manifest_path) + ".sha256")
    if not sha_path.exists():
        raise Phase12Error("Frozen ground-truth checksum is missing.")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise Phase12Error("Frozen ground-truth manifest checksum mismatch.")

    payload = read_json(manifest_path)
    if payload.get("release_status") != "frozen":
        raise Phase12Error("Human ground truth is not frozen.")
    if payload.get("generation_intent_used_as_ground_truth") is not False:
        raise Phase12Error("Generation intent must not be human ground truth.")

    for rel_path, expected_sha in payload["file_sha256"].items():
        path = root / rel_path
        if not path.exists() or sha256_file(path) != expected_sha:
            raise Phase12Error("Frozen human ground-truth artifact changed: {}".format(rel_path))
    return payload

def _active_design_info(root: Path) -> Dict[str, Any]:
    root = Path(root)
    manifest = validate_design_freeze(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1" / "frozen_design"
    path = base / "design_manifest.json"
    return {"manifest": manifest, "sha256": sha256_file(path)}


def _selector_run_metadata(root: Path) -> Dict[str, Any]:
    phase10 = phase10_protocol_info(root)
    phase11 = phase11_selector_info(root)
    return {
        "phase10_protocol_sha256": phase10["sha256"],
        "phase11_selector_manifest_sha256": phase11["sha256"],
        "required_model": phase10["journal_llm"]["required_model_tag"],
        "prompt_sha256": phase10["journal_llm"]["prompt_sha256"],
        "schema_sha256": phase10["journal_llm"]["schema_sha256"],
        "runtime_lock": phase10["journal_llm"]["runtime_lock"],
        "generation": phase10["journal_llm"]["generation"],
    }


def run_main_benchmark(root: Path, *, resume: bool = False) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    validate_ground_truth_freeze(root)
    gt_info = validate_ground_truth(root)
    ground_truth = gt_info["rows"]

    from awareml.llm.journal_client import JournalModelLockError
    from awareml.llm.objective_selection import JournalObjectiveSelector

    selector = JournalObjectiveSelector(root=root)
    runtime = selector.client.verify_runtime()
    meta = _selector_run_metadata(root)

    output = (
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "results"
        / "llm_outputs.jsonl"
    )

    existing: Dict[str, Dict[str, Any]] = {}
    if output.exists():
        if not resume:
            raise Phase12Error(
                "{} already exists. Use --resume only after an interrupted first run; "
                "do not silently rerun completed cases.".format(output)
            )
        for row in read_jsonl(output):
            existing[row["scenario_id"]] = row

    rows: List[Dict[str, Any]] = list(existing.values())
    complete_ids = set(existing)
    started = utc_now()

    for index, gt in enumerate(ground_truth, start=1):
        sid = gt["scenario_id"]
        if sid in complete_ids:
            continue
        t0 = time.perf_counter()
        try:
            result = selector.select(gt["scenario"])
        except JournalModelLockError:
            raise
        latency = time.perf_counter() - t0
        row = {
            "scenario_id": sid,
            "scenario": gt["scenario"],
            "status": result.status,
            "selected_objectives": list(result.selected_objectives),
            "uncertainties": list(result.uncertainties),
            "source": result.source,
            "model": result.model,
            "fallback_used": bool(result.fallback_used),
            "latency_sec": latency,
            "attempt_index": 1,
            "benchmark_order": index,
        }
        rows.append(row)
        rows.sort(key=lambda x: int(x["benchmark_order"]))
        write_jsonl(output, rows)

    if len(rows) != len(ground_truth):
        raise Phase12Error("Main benchmark did not produce one output per ground-truth case.")

    metadata = {
        "artifact": "phase12_main_objective_selection_run",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "n": len(rows),
        "runtime_verified": runtime,
        "frozen_inputs": meta,
        "design_manifest_sha256": _active_design_info(root)["sha256"],
        "fallback_allowed": False,
        "attempts_per_case": 1,
    }
    write_json(output.with_name("llm_run_metadata.json"), metadata)
    return {"outputs": rows, "metadata": metadata}


def evaluate_main_benchmark(root: Path) -> Dict[str, Any]:
    root = Path(root)
    gt = validate_ground_truth(root)["rows"]
    results = (
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "results"
    )
    output_path = results / "llm_outputs.jsonl"
    if not output_path.exists():
        raise Phase12Error("Run the frozen LLaMA benchmark first.")
    predictions = read_jsonl(output_path)
    metrics = evaluate_predictions(gt, predictions)
    metrics.update(
        {
            "artifact": "phase12_objective_selection_metrics",
            "evaluated_model": _selector_run_metadata(root)["required_model"],
            "ground_truth_source": "independent_human_majority_vote",
            "hard_cases_excluded_from_primary_metrics": True,
        }
    )
    write_json(results / "metrics.json", metrics)
    return metrics


def run_paraphrase_benchmark(root: Path, *, resume: bool = False) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    validate_ground_truth_freeze(root)
    review = validate_paraphrase_review(root)

    from awareml.llm.journal_client import JournalModelLockError
    from awareml.llm.objective_selection import JournalObjectiveSelector

    selector = JournalObjectiveSelector(root=root)
    runtime = selector.client.verify_runtime()

    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    design_rows = read_csv(base / "design" / "paraphrase_families.generated.csv")
    output = base / "results" / "paraphrase_llm_outputs.jsonl"

    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if output.exists():
        if not resume:
            raise Phase12Error(
                "{} already exists. Use --resume only after interruption.".format(output)
            )
        for row in read_jsonl(output):
            existing[(row["family_id"], row["variant_id"])] = row

    rows = list(existing.values())
    order = 0
    for row in design_rows:
        order += 1
        key = (row["family_id"], row["variant_id"])
        if key in existing:
            continue
        t0 = time.perf_counter()
        try:
            result = selector.select(row["scenario"])
        except JournalModelLockError:
            raise
        rows.append(
            {
                "family_id": row["family_id"],
                "variant_id": row["variant_id"],
                "variant_type": row["variant_type"],
                "scenario": row["scenario"],
                "target_objectives": review["targets"][row["family_id"]],
                "status": result.status,
                "selected_objectives": list(result.selected_objectives),
                "uncertainties": list(result.uncertainties),
                "model": result.model,
                "fallback_used": bool(result.fallback_used),
                "latency_sec": time.perf_counter() - t0,
                "benchmark_order": order,
                "attempt_index": 1,
            }
        )
        rows.sort(key=lambda x: int(x["benchmark_order"]))
        write_jsonl(output, rows)

    if len(rows) != 60:
        raise Phase12Error("Paraphrase benchmark must contain 60 total outputs.")

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family_id"], []).append(row)

    base_exact = 0
    para_exact = 0
    para_consistent = 0
    para_count = 0
    family_reports = []

    for family_id in sorted(by_family):
        fam_rows = by_family[family_id]
        base_row = [r for r in fam_rows if r["variant_type"] == "base"][0]
        paras = [r for r in fam_rows if r["variant_type"] == "paraphrase"]
        target = set(review["targets"][family_id])
        base_pred = set(base_row["selected_objectives"])
        base_ok = base_pred == target
        base_exact += int(base_ok)

        correct = 0
        consistent = 0
        for r in paras:
            pred = set(r["selected_objectives"])
            correct += int(pred == target)
            consistent += int(pred == base_pred)
            para_count += 1
        para_exact += correct
        para_consistent += consistent

        family_reports.append(
            {
                "family_id": family_id,
                "target_objectives": sorted(target, key=OBJECTIVES.index),
                "base_selected_objectives": base_row["selected_objectives"],
                "base_exact": base_ok,
                "paraphrase_exact_count": correct,
                "paraphrase_consistent_with_base_count": consistent,
                "paraphrase_count": len(paras),
            }
        )

    report = {
        "artifact": "phase12_paraphrase_robustness",
        "families": 10,
        "paraphrase_evaluations": para_count,
        "base_exact_match_rate": base_exact / 10.0,
        "paraphrase_exact_match_rate": para_exact / float(para_count),
        "paraphrase_prediction_consistency_rate": para_consistent / float(para_count),
        "set_change_rate": 1.0 - (para_consistent / float(para_count)),
        "human_reviewers": review["reviewer_ids"],
        "family_results": family_reports,
        "runtime_verified": runtime,
    }
    write_json(base / "results" / "paraphrase_metrics.json", report)
    return report


def run_adversarial_benchmark(root: Path, *, resume: bool = False) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    validate_ground_truth_freeze(root)

    from awareml.llm.journal_client import JournalModelLockError
    from awareml.llm.objective_selection import JournalObjectiveSelector

    selector = JournalObjectiveSelector(root=root)
    runtime = selector.client.verify_runtime()

    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    cases = read_csv(base / "design" / "adversarial_set.csv")
    output = base / "results" / "adversarial_llm_outputs.jsonl"

    existing = {}
    if output.exists():
        if not resume:
            raise Phase12Error(
                "{} already exists. Use --resume only after interruption.".format(output)
            )
        for row in read_jsonl(output):
            existing[row["case_id"]] = row

    rows = list(existing.values())
    order = 0
    for case in cases:
        order += 1
        if case["case_id"] in existing:
            continue
        t0 = time.perf_counter()
        try:
            result = selector.select(case["scenario"])
        except JournalModelLockError:
            raise
        taxonomy = classify_adversarial_case(
            expected_status=case["expected_status"],
            expected_behavior=case["expected_behavior"],
            actual_status=result.status,
            selected_objectives=result.selected_objectives,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "expected_status": case["expected_status"],
                "expected_behavior": case["expected_behavior"],
                "actual_status": result.status,
                "selected_objectives": list(result.selected_objectives),
                "uncertainties": list(result.uncertainties),
                "taxonomy": taxonomy,
                "model": result.model,
                "fallback_used": bool(result.fallback_used),
                "latency_sec": time.perf_counter() - t0,
                "benchmark_order": order,
                "attempt_index": 1,
            }
        )
        rows.sort(key=lambda x: int(x["benchmark_order"]))
        write_jsonl(output, rows)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["taxonomy"]] = counts.get(row["taxonomy"], 0) + 1

    report = {
        "artifact": "phase12_adversarial_failure_taxonomy",
        "n": len(rows),
        "taxonomy_counts": counts,
        "cases": rows,
        "runtime_verified": runtime,
    }
    write_json(base / "results" / "adversarial_failure_taxonomy.json", report)
    return report


def final_artifact_paths(root: Path) -> List[Path]:
    root = Path(root)
    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    return [
        base / "frozen_design" / "design_manifest.json",
        base / "frozen_ground_truth" / "ground_truth_manifest.json",
        base / "results" / "ground_truth.jsonl",
        base / "results" / "hard_cases.annotation.jsonl",
        base / "results" / "annotation_agreement.json",
        base / "results" / "llm_outputs.jsonl",
        base / "results" / "llm_run_metadata.json",
        base / "results" / "metrics.json",
        base / "results" / "paraphrase_llm_outputs.jsonl",
        base / "results" / "paraphrase_metrics.json",
        base / "results" / "adversarial_llm_outputs.jsonl",
        base / "results" / "adversarial_failure_taxonomy.json",
        base / "human" / "final_annotation_pool.csv",
        base / "human" / "paraphrase_family_review.csv",
    ]


def freeze_benchmark(root: Path) -> Dict[str, Any]:
    root = Path(root)
    validate_design_freeze(root)
    gt = validate_ground_truth(root)
    review = validate_paraphrase_review(root)

    base = root / "data" / "journal" / "objective_selection_benchmark_v1"
    metrics = read_json(base / "results" / "metrics.json")
    paraphrase = read_json(base / "results" / "paraphrase_metrics.json")
    adversarial = read_json(base / "results" / "adversarial_failure_taxonomy.json")

    paths = final_artifact_paths(root)
    for path in paths:
        if not path.exists():
            raise Phase12Error("Required Phase-12 final artifact missing: {}".format(path))

    file_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in paths
    }

    phase10 = phase10_protocol_info(root)
    phase11 = phase11_selector_info(root)
    design = _active_design_info(root)
    ground_truth_freeze = validate_ground_truth_freeze(root)
    ground_truth_manifest_path = (
        base / "frozen_ground_truth" / "ground_truth_manifest.json"
    )

    manifest = {
        "artifact": "objective_selection_benchmark_v1",
        "release_status": "frozen",
        "frozen_at_utc": utc_now(),
        "primary_benchmark_n": gt["count"],
        "annotation_agreement": gt["agreement"],
        "metrics": metrics,
        "paraphrase_summary": {
            "families": paraphrase["families"],
            "paraphrase_evaluations": paraphrase["paraphrase_evaluations"],
            "paraphrase_exact_match_rate": paraphrase["paraphrase_exact_match_rate"],
            "paraphrase_prediction_consistency_rate": paraphrase[
                "paraphrase_prediction_consistency_rate"
            ],
        },
        "adversarial_summary": {
            "n": adversarial["n"],
            "taxonomy_counts": adversarial["taxonomy_counts"],
        },
        "phase10_protocol_sha256": phase10["sha256"],
        "phase11_selector_manifest_sha256": phase11["sha256"],
        "phase12_design_manifest_sha256": design["sha256"],
        "phase12_ground_truth_manifest_sha256": sha256_file(
            ground_truth_manifest_path
        ),
        "journal_model": phase10["journal_llm"]["required_model_tag"],
        "objective_vocabulary": list(OBJECTIVES),
        "generation_intent_used_as_ground_truth": False,
        "human_ground_truth": True,
        "human_paraphrase_semantics_reviewed": True,
        "heldout_dataset_contents_used": False,
        "file_sha256": file_hashes,
    }

    frozen = base / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    manifest_path = frozen / "manifest.json"
    write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    (frozen / "manifest.json.sha256").write_text(
        "{}  manifest.json\n".format(digest), encoding="utf-8"
    )
    (root / "data" / "journal" / "active_objective_benchmark.txt").write_text(
        "objective_selection_benchmark_v1/frozen/manifest.json\n",
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "sha256": digest, "payload": manifest}


def validate_complete(root: Path) -> Dict[str, Any]:
    root = Path(root)
    marker = root / "data" / "journal" / "active_objective_benchmark.txt"
    if not marker.exists():
        raise Phase12Error("Phase-12 active benchmark marker is missing.")
    rel = marker.read_text(encoding="utf-8").strip()
    manifest_path = root / "data" / "journal" / rel
    if not manifest_path.exists():
        raise Phase12Error("Frozen Phase-12 benchmark manifest is missing.")

    sha_path = Path(str(manifest_path) + ".sha256")
    if not sha_path.exists():
        raise Phase12Error("Frozen Phase-12 benchmark checksum is missing.")
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise Phase12Error("Frozen Phase-12 benchmark manifest checksum mismatch.")

    manifest = read_json(manifest_path)
    if manifest.get("release_status") != "frozen":
        raise Phase12Error("Phase-12 benchmark is not frozen.")
    if manifest.get("heldout_dataset_contents_used") is not False:
        raise Phase12Error("Held-out dataset policy violation.")
    if manifest.get("generation_intent_used_as_ground_truth") is not False:
        raise Phase12Error("Generation intent must never become ground truth.")

    for rel_path, expected_sha in manifest["file_sha256"].items():
        path = root / rel_path
        if not path.exists() or sha256_file(path) != expected_sha:
            raise Phase12Error("Frozen Phase-12 artifact changed: {}".format(rel_path))

    gt = read_jsonl(
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "results"
        / "ground_truth.jsonl"
    )
    outputs = read_jsonl(
        root
        / "data"
        / "journal"
        / "objective_selection_benchmark_v1"
        / "results"
        / "llm_outputs.jsonl"
    )
    if len(gt) != len(outputs):
        raise Phase12Error("Ground truth and LLM output counts differ.")

    return {
        "sha256": actual,
        "manifest": manifest,
        "primary_n": len(gt),
    }
