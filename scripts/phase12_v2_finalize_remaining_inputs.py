from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from awareml.journal.objective_benchmark_v2 import (
    OBJECTIVES,
    OBJECTIVE_COLUMNS,
    Phase12V2Error,
    freeze_adversarial_set,
    freeze_paraphrase_set,
    labels_from_pipe,
    labels_to_pipe,
    objective_set_from_columns,
    parse_bool,
    prepare_paraphrase_reviews,
    read_csv,
    read_json,
    sha256_file,
    validate_all_evaluation_inputs_frozen,
    validate_ground_truth_freeze,
    verify_selector_freezes,
    write_csv,
    write_json,
)

BASE = ROOT / "data" / "journal" / "objective_selection_benchmark_v2"
AMENDED_PRIMARY = BASE / "results" / "primary_benchmark_amended.csv"
AMENDED_MANIFEST = BASE / "frozen" / "amended_primary_manifest.json"
PRIMARY = BASE / "frozen" / "primary_benchmark.csv"
PRIMARY_MANIFEST = BASE / "frozen" / "primary_manifest.json"
PRIMARY_MANIFEST_SHA = Path(str(PRIMARY_MANIFEST) + ".sha256")
PARA_DIR = BASE / "paraphrases"
ADV_DIR = BASE / "adversarial"

EXPECTED_PRIMARY_N = 60
EXPECTED_K = {"1":15, "2":15, "3":15, "4":15}
MIN_HUMAN = 24
EXPECTED_PARA_FAMILIES = 10
EXPECTED_PARA_VARIANTS = 50

def _yes(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes","true","1","y"}

def _gt_from_row(row):
    raw = str(row.get("ground_truth","")).strip()
    if raw:
        labels = labels_from_pipe(raw)
    else:
        labels = [
            obj for obj, col in zip(OBJECTIVES, OBJECTIVE_COLUMNS)
            if _yes(row.get(col,""))
        ]
    return labels_to_pipe(labels)

def _sha_text(path: Path, name: str):
    digest = sha256_file(path)
    Path(str(path)+".sha256").write_text(f"{digest}  {name}\n", encoding="utf-8")
    return digest

def activate_primary():
    validate_ground_truth_freeze(ROOT)
    verify_selector_freezes(ROOT)

    if not AMENDED_PRIMARY.exists() or not AMENDED_MANIFEST.exists():
        raise Phase12V2Error("Missing frozen amended primary artifacts.")

    amended_manifest = read_json(AMENDED_MANIFEST)
    expected_sha = amended_manifest.get("selection",{}).get("primary_csv_sha256")
    actual_sha = sha256_file(AMENDED_PRIMARY)
    if expected_sha != actual_sha:
        raise Phase12V2Error("primary_benchmark_amended.csv no longer matches its frozen amended manifest.")

    rows = read_csv(AMENDED_PRIMARY)
    if len(rows) != EXPECTED_PRIMARY_N:
        raise Phase12V2Error(f"Expected 60 amended-primary rows; found {len(rows)}.")

    materialized = []
    kcounts = Counter()
    source_counts = Counter()
    gt_counts = Counter()
    for idx, row in enumerate(rows, 1):
        sid = str(row.get("scenario_id","")).strip()
        scenario = str(row.get("scenario","")).strip()
        if not sid or not scenario:
            raise Phase12V2Error("Every primary row needs scenario_id and scenario.")

        gt = _gt_from_row(row)
        labels = labels_from_pipe(gt)
        k = int(str(row.get("k_prime","0")).strip())
        if len(labels) != k:
            raise Phase12V2Error(
                f"{sid}: human-ground-truth set {gt!r} has {len(labels)} labels but k_prime={k}."
            )
        source = str(row.get("source_type") or row.get("source") or "").strip()
        if source not in {"human","generated"}:
            raise Phase12V2Error(f"{sid}: invalid source type {source!r}.")

        kcounts[str(k)] += 1
        source_counts[source] += 1
        gt_counts[gt] += 1

        materialized.append({
            "benchmark_id":f"V2-{idx:03d}",
            "scenario_id":sid,
            "scenario":scenario,
            "ground_truth":gt,
            "k_prime":k,
            "source":source,
            "domain":row.get("domain",""),
            "style":row.get("style",""),
            "pairwise_jaccard":row.get("pairwise_jaccard",""),
            "vote_accuracy":row.get("vote_accuracy",""),
            "vote_runtime":row.get("vote_runtime",""),
            "vote_energy":row.get("vote_energy",""),
            "vote_co2":row.get("vote_co2",""),
        })

    if dict(kcounts) != EXPECTED_K:
        raise Phase12V2Error(f"Expected k' balance {EXPECTED_K}; found {dict(kcounts)}.")
    if source_counts["human"] < MIN_HUMAN:
        raise Phase12V2Error(f"Need at least {MIN_HUMAN} human cases; found {source_counts['human']}.")
    if len(gt_counts) < EXPECTED_PARA_FAMILIES:
        raise Phase12V2Error("Primary set does not contain 10 distinct objective sets.")

    if PRIMARY.exists() or PRIMARY_MANIFEST.exists() or PRIMARY_MANIFEST_SHA.exists():
        # Idempotent only if a valid bridge already exists.
        if not (PRIMARY.exists() and PRIMARY_MANIFEST.exists() and PRIMARY_MANIFEST_SHA.exists()):
            raise Phase12V2Error("Partial canonical primary freeze exists; inspect manually.")
        payload = read_json(PRIMARY_MANIFEST)
        expected = PRIMARY_MANIFEST_SHA.read_text(encoding="utf-8").split()[0]
        if sha256_file(PRIMARY_MANIFEST) != expected:
            raise Phase12V2Error("Existing primary_manifest checksum mismatch.")
        for rel, digest in payload.get("file_sha256",{}).items():
            p = ROOT / rel
            if not p.exists() or sha256_file(p) != digest:
                raise Phase12V2Error(f"Existing canonical primary asset changed: {rel}")
        return {"status":"ALREADY_ACTIVE_AND_VALID","manifest_sha256":expected,"summary":payload.get("summary")}

    write_csv(
        PRIMARY,
        materialized,
        ["benchmark_id","scenario_id","scenario","ground_truth","k_prime","source",
         "domain","style","pairwise_jaccard","vote_accuracy","vote_runtime","vote_energy","vote_co2"],
    )

    selector_freezes = verify_selector_freezes(ROOT)
    ground_truth_manifest = BASE / "frozen" / "ground_truth_manifest.json"
    file_hashes = {
        str(PRIMARY.relative_to(ROOT)).replace("\\","/"):sha256_file(PRIMARY),
        str(AMENDED_PRIMARY.relative_to(ROOT)).replace("\\","/"):sha256_file(AMENDED_PRIMARY),
        str(AMENDED_MANIFEST.relative_to(ROOT)).replace("\\","/"):sha256_file(AMENDED_MANIFEST),
        str(ground_truth_manifest.relative_to(ROOT)).replace("\\","/"):sha256_file(ground_truth_manifest),
        selector_freezes["legacy_v2"]["path"]:selector_freezes["legacy_v2"]["sha256"],
        selector_freezes["v32"]["path"]:selector_freezes["v32"]["sha256"],
    }
    summary = {
        "n":len(materialized),
        "k_prime_counts":dict(kcounts),
        "human_written":source_counts["human"],
        "generated":source_counts["generated"],
        "human_written_fraction":source_counts["human"]/float(len(materialized)),
        "objective_set_counts":dict(gt_counts),
        "distinct_objective_sets":len(gt_counts),
        "selection_used_model_outputs":False,
        "selection_used_generator_intent":False,
        "materialization_note":(
            "Selection/order are inherited unchanged from frozen amended_primary_manifest. "
            "For amendment rows whose convenience CSV ground_truth field was blank, the pipe-delimited "
            "ground-truth set is deterministically materialized from the frozen per-objective majority-vote "
            "yes/no columns; k_prime is checked for consistency."
        ),
    }
    manifest = {
        "artifact":"objective_selection_benchmark_v2_primary_amended_materialization",
        "release_status":"frozen",
        "parent_amended_primary_manifest_sha256":sha256_file(AMENDED_MANIFEST),
        "summary":summary,
        "ground_truth_source":"independent_human_majority_vote_only, including frozen H041-H055 amendment",
        "generator_intent_used_as_ground_truth":False,
        "evaluated_selector_outputs_existed_before_primary_materialization":False,
        "selection_algorithm":"inherited exactly from frozen amended_primary_manifest; no reselection",
        "file_sha256":file_hashes,
    }
    write_json(PRIMARY_MANIFEST, manifest)
    digest = _sha_text(PRIMARY_MANIFEST, "primary_manifest.json")
    return {"status":"FROZEN","manifest_sha256":digest,"summary":summary}

def sync_and_validate_paraphrases():
    if not PRIMARY.exists() or not PRIMARY_MANIFEST.exists():
        raise Phase12V2Error("Run activate-primary first.")

    primary = {r["scenario_id"]:r for r in read_csv(PRIMARY)}
    bases = read_csv(PARA_DIR/"base_scenarios.csv")
    candidates = read_csv(PARA_DIR/"paraphrase_candidates.csv")
    if len(bases) != EXPECTED_PARA_FAMILIES:
        raise Phase12V2Error(f"Expected 10 paraphrase bases; found {len(bases)}.")
    if len(candidates) != EXPECTED_PARA_VARIANTS:
        raise Phase12V2Error(f"Expected 50 paraphrase candidates; found {len(candidates)}.")

    seen_gt=set()
    for row in bases:
        sid=row["scenario_id"].strip()
        if sid not in primary:
            raise Phase12V2Error(f"Paraphrase base {sid} is not in canonical primary.")
        p=primary[sid]
        if row["base_scenario"].strip()!=p["scenario"].strip():
            raise Phase12V2Error(f"{sid}: base text differs from frozen primary.")
        if row["ground_truth"].strip()!=p["ground_truth"].strip():
            raise Phase12V2Error(f"{sid}: base ground truth differs from frozen primary.")
        if int(row["k_prime"])!=int(p["k_prime"]):
            raise Phase12V2Error(f"{sid}: base k_prime differs from frozen primary.")
        row["benchmark_id"]=p["benchmark_id"]
        seen_gt.add(row["ground_truth"].strip())

    if len(seen_gt) != EXPECTED_PARA_FAMILIES:
        raise Phase12V2Error("The 10 paraphrase families must have 10 distinct ground-truth sets.")

    write_csv(
        PARA_DIR/"base_scenarios.csv",
        bases,
        ["family_id","benchmark_id","scenario_id","base_scenario","ground_truth","k_prime"],
    )

    keys=set()
    for row in candidates:
        key=(row["family_id"].strip(),row["variant_id"].strip())
        if key in keys:
            raise Phase12V2Error(f"Duplicate paraphrase candidate key {key}.")
        keys.add(key)
        scenario=row["scenario"].strip()
        if not scenario:
            raise Phase12V2Error(f"Blank paraphrase {key}.")
        model=row["generator_model"].strip()
        if not model or model.lower()=="llama3:8b":
            raise Phase12V2Error(f"{key}: paraphrase generator must be external to evaluated llama3:8b.")
        lower=" "+scenario.lower().replace("co₂","co2")+" "
        for forbidden in (" accuracy "," runtime "," energy "," co2 "):
            if forbidden in lower:
                raise Phase12V2Error(f"{key}: literal frozen objective name leaked into paraphrase: {forbidden.strip()}")

    # Regenerate blinded review packets in the repository-native deterministic order.
    result = prepare_paraphrase_reviews(ROOT, force=True)

    # Create a reviewer reference that excludes hidden ground truth.
    ref_rows=[{"family_id":r["family_id"],"base_scenario":r["base_scenario"]} for r in bases]
    write_csv(PARA_DIR/"review_reference.csv",ref_rows,["family_id","base_scenario"])
    return {
        "status":"READY_FOR_TWO_INDEPENDENT_REVIEWERS",
        "families":len(bases),
        "distinct_ground_truth_sets":len(seen_gt),
        "paraphrase_candidates":len(candidates),
        "native_review_packets":result["review_packets"],
        "send_with_each_packet":[
            str((PARA_DIR/"review_reference.csv").relative_to(ROOT)).replace("\\","/"),
            str((PARA_DIR/"PARAPHRASE_REVIEW_INSTRUCTIONS.txt").relative_to(ROOT)).replace("\\","/"),
        ],
    }

def freeze_adversarial():
    rows=read_csv(ADV_DIR/"adversarial_cases.csv")
    if len(rows)!=12:
        raise Phase12V2Error(f"Expected 12 adversarial cases in this package; found {len(rows)}.")
    counts=Counter(r["expected_status"].strip() for r in rows)
    if counts != Counter({"ambiguous":4,"contradictory":4,"out_of_scope":4}):
        raise Phase12V2Error(f"Unexpected adversarial status balance: {dict(counts)}")
    return freeze_adversarial_set(ROOT)

def validate_paraphrase_reviews():
    bases={r["family_id"]:r for r in read_csv(PARA_DIR/"base_scenarios.csv")}
    cand={(r["family_id"],r["variant_id"]):r for r in read_csv(PARA_DIR/"paraphrase_candidates.csv")}
    packets=[]
    reviewer_ids=[]
    problems=[]
    for letter in ("A","B"):
        path=PARA_DIR/f"review_{letter}.csv"
        rows=read_csv(path)
        rowmap={(r["family_id"],r["variant_id"]):r for r in rows}
        if set(rowmap)!=set(cand):
            raise Phase12V2Error(f"{path.name} does not match the 50 candidate variants.")
        ids={r.get("reviewer_id","").strip() for r in rows}
        if "" in ids or len(ids)!=1:
            raise Phase12V2Error(f"{path.name} needs one nonblank reviewer_id on every row.")
        reviewer_ids.append(next(iter(ids)))
        packets.append(rowmap)
    if len(set(reviewer_ids))!=2:
        raise Phase12V2Error("Two distinct paraphrase reviewer IDs are required.")

    for key in sorted(cand):
        fam,_=key
        expected=set(labels_from_pipe(bases[fam]["ground_truth"]))
        for label,packet in zip(("A","B"),packets):
            row=packet[key]
            preserved=bool(parse_bool(row.get("meaning_preserved"),allow_blank=False))
            recovered=objective_set_from_columns(row)
            if not preserved:
                problems.append({"family_id":key[0],"variant_id":key[1],"reviewer":label,"reason":"meaning_not_preserved"})
            if recovered!=expected:
                problems.append({
                    "family_id":key[0],"variant_id":key[1],"reviewer":label,
                    "reason":"objective_set_mismatch",
                    "expected":sorted(expected),"recovered":sorted(recovered),
                })
    if problems:
        raise Phase12V2Error(
            "Paraphrase review is not freeze-ready. Do NOT alter reviewer judgments. "
            f"{len(problems)} review issue(s) require replacement/regeneration before freezing. "
            "See docs/PHASE12_V2_REMAINING_INPUTS_FINALIZATION.md."
        )
    return {
        "status":"PASS_READY_TO_FREEZE_PARAPHRASES",
        "reviewers":reviewer_ids,
        "families":len(bases),
        "variants":len(cand),
        "all_50_unanimously_meaning_preserved_and_ground_truth_recovered":True,
    }

def freeze_paraphrases_strict():
    validate_paraphrase_reviews()
    result=freeze_paraphrase_set(ROOT)
    accepted=int(result.get("summary",{}).get("accepted_variants",0))
    if accepted!=EXPECTED_PARA_VARIANTS:
        raise Phase12V2Error(f"Strict freeze expected all 50 accepted; native freeze reports {accepted}.")
    return result

def status():
    payload={
        "canonical_primary":PRIMARY.exists(),
        "canonical_primary_manifest":PRIMARY_MANIFEST.exists(),
        "paraphrase_manifest":(BASE/"frozen/paraphrase_manifest.json").exists(),
        "adversarial_manifest":(BASE/"frozen/adversarial_manifest.json").exists(),
        "final_manifest":(BASE/"frozen/manifest.json").exists(),
        "model_outputs_present":any((BASE/"results"/n).exists() for n in (
            "baseline_primary_outputs.jsonl","v32_primary_outputs.jsonl",
            "baseline_paraphrase_outputs.jsonl","v32_paraphrase_outputs.jsonl",
            "baseline_adversarial_outputs.jsonl","v32_adversarial_outputs.jsonl",
        )),
    }
    if payload["canonical_primary"]:
        rows=read_csv(PRIMARY)
        payload["primary_n"]=len(rows)
        payload["primary_k_counts"]=dict(Counter(r["k_prime"] for r in rows))
        payload["primary_source_counts"]=dict(Counter(r["source"] for r in rows))
        payload["primary_distinct_ground_truth_sets"]=len(set(r["ground_truth"] for r in rows))
    return payload

def main():
    ap=argparse.ArgumentParser(description="Finalize remaining Phase-12-v2 pre-evaluation inputs")
    ap.add_argument("command", choices=[
        "status","activate-primary","prepare-inputs","freeze-adversarial",
        "validate-paraphrase-reviews","freeze-paraphrases","validate-evaluation-freeze",
    ])
    args=ap.parse_args()
    try:
        if args.command=="status":
            out=status()
        elif args.command=="activate-primary":
            out=activate_primary()
        elif args.command=="prepare-inputs":
            out={"paraphrases":sync_and_validate_paraphrases(),"adversarial":freeze_adversarial()}
        elif args.command=="freeze-adversarial":
            out=freeze_adversarial()
        elif args.command=="validate-paraphrase-reviews":
            out=validate_paraphrase_reviews()
        elif args.command=="freeze-paraphrases":
            out=freeze_paraphrases_strict()
        elif args.command=="validate-evaluation-freeze":
            out=validate_all_evaluation_inputs_frozen(ROOT)
        else:
            raise AssertionError(args.command)
        print(json.dumps(out,indent=2,ensure_ascii=False,sort_keys=True))
        return 0
    except Phase12V2Error as exc:
        print("PHASE12-V2 FINALIZATION ERROR:",exc,file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
