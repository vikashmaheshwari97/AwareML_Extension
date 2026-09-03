# Phase 12 — NL Objective-Selection Benchmark

Phase 12 evaluates **Problem A only**:

> Given a natural-language streaming AutoML deployment scenario, which subset of
> the frozen objectives `{Accuracy, Runtime, Energy, CO2}` is implied?

It does **not** evaluate framework selection and it does **not** use the 705
meta-logs as Copilot training data.

## Frozen separation

- **External scenario generator:** GPT-5.6 Sol.
- **Model under evaluation:** exact Phase-10 `llama3:8b`.
- **Prompt/schema/runtime:** inherited from Journal Experimental Protocol v1.
- **Weighting policy:** irrelevant to the primary selection metrics.
- **Ground truth:** independent human annotation only.
- **Generation intent:** coverage/design metadata only; never scoring truth.
- **23 held-out datasets:** untouched.

## 12.1–12.2 Generated candidate pool

The bundle contains **120 generated candidate scenarios**, balanced:

- k'=1: 30
- k'=2: 30
- k'=3: 30
- k'=4: 30

The public candidate file intentionally contains no hidden objective labels.
A separate `*.PRIVATE.csv` file records generation intent **only to audit design
coverage**. Do not show that private file to filterers or annotators.

The generated statements do not explicitly use the frozen words
`Accuracy`, `Runtime`, `Energy`, `CO2/CO₂`.

## 12.3 Human realism filtering

Give an uninvolved colleague/student:

`data/journal/objective_selection_benchmark_v1/human/realism_filter.csv`

Ask them to fill:

- `keep`: yes/no
- `realism_1_to_5`
- `ambiguous_or_unclear`: yes/no when useful
- `notes`
- `filterer_id`: anonymous study code, not a name

Retain **40–50 generated scenarios**.

Optionally add **0–15 genuine human-written scenarios** in:

`human_written_scenarios.csv`

The final annotation pool must contain **50–65 statements**.

## 12.4 Independent ground-truth annotation

After realism filtering:

```powershell
python .\scripts\prepare_phase12_annotation_packets.py
```

This creates:

- `annotations_A.csv`
- `annotations_B.csv`
- `annotations_C.csv`

Give each file to a different annotator. Do not share other annotators' labels
and do not show the private generation-intent file.

Each annotator checks:

- Accuracy
- Runtime
- Energy
- CO2
- Ambiguous

Use 1/0 or yes/no.

When 2–3 independent files are complete:

```powershell
python .\scripts\aggregate_phase12_annotations.py
```

Then, **before any evaluated-LLaMA run**, freeze the resulting human ground truth:

```powershell
python .\scripts\freeze_phase12_ground_truth.py
```

This hashes the raw completed annotator files, filtering decisions, agreement
report, resolved ground truth, hard cases and paraphrase review. The LLaMA
benchmark runners refuse to start without this freeze.

Phase 12 computes:

- per-objective Fleiss kappa;
- mean defined kappa;
- unanimous full-set agreement;
- mean pairwise Jaccard;
- majority-vote ground truth;
- unresolved/hard cases.

Ambiguous/tied cases are not forced into artificial ground truth. They are
saved separately and excluded from primary metrics.

## Paraphrase human review

Before running the paraphrase benchmark, complete:

`human/paraphrase_family_review.csv`

For each of the 10 families:

1. label the intended objective subset for the base scenario;
2. confirm that **all five paraphrases preserve the same meaning**;
3. provide an anonymous reviewer ID.

This protects the robustness test from using model-generated intent as truth.

## 12.5 Run exact journal LLaMA

Only after the design is frozen and human ground truth exists:

```powershell
python .\scripts\run_phase12_objective_benchmark.py
```

The script uses `JournalObjectiveSelector`, which enforces:

- exact `llama3:8b`;
- exact frozen model digest;
- exact Ollama version;
- exact production prompt;
- exact response schema;
- no model fallback.

Each case receives one benchmark attempt. `--resume` only fills missing cases
after an interrupted run; it does not rerun completed cases.

## 12.6 Metrics

```powershell
python .\scripts\evaluate_phase12_objective_benchmark.py
```

Outputs:

- precision/recall/F1 for each objective;
- micro precision/recall/F1;
- macro-F1;
- exact-match rate;
- results by k'=1,2,3,4;
- prediction-status counts.

## 12.7 Paraphrase robustness

The design contains:

- 10 base scenarios;
- 5 meaning-preserving paraphrases per base;
- 50 paraphrase evaluations.

Run:

```powershell
python .\scripts\run_phase12_paraphrase_benchmark.py
```

Reports:

- base exact-match rate;
- paraphrase exact-match rate;
- prediction consistency with base;
- set-change rate.

## 12.8 Adversarial evaluation

The frozen design contains 14 adversarial cases covering:

- sensible default / ambiguity;
- over-selection risk;
- contradiction handling;
- out-of-scope handling;
- silent/malformed failure.

Run:

```powershell
python .\scripts\run_phase12_adversarial_benchmark.py
```

## Two freezes

Phase 12 deliberately has three freezes.

### Design freeze

Run **before** human annotation and before any benchmark LLaMA outputs:

```powershell
python .\scripts\freeze_phase12_design.py
```

This prevents changing the scenario pool after seeing LLaMA performance.

### Human ground-truth freeze

Run after annotation aggregation and paraphrase review, but before any LLaMA
evaluation:

```powershell
python .\scripts\freeze_phase12_ground_truth.py
```

### Final benchmark freeze

After the frozen human ground truth, main metrics, paraphrase results, and
adversarial results exist:

```powershell
python .\scripts\freeze_phase12_benchmark.py
python .\scripts\validate_phase12_complete.py
```

The final artifact is:

`objective_selection_benchmark_v1`

with a SHA256 manifest.

## Important scientific rule

Do not modify Phase-11 objective-selection logic after inspecting Phase-12
benchmark results. Any future selector change becomes a new selector/version
and requires a new evaluation protocol.
