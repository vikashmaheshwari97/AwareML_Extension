# Phase-12-v2 Remaining Input Finalization

This package finishes the two pre-evaluation benchmark inputs that remain after the
amended primary 60 was frozen: the 10×5 paraphrase test and the fresh adversarial set.

It does **not** run either evaluated selector before those inputs are frozen.

## Why a canonical primary materialization is needed

The research-preserving amendment froze:
- `results/primary_benchmark_amended.csv`
- `frozen/amended_primary_manifest.json`

The repository-native Phase-12-v2 pipeline expects:
- `frozen/primary_benchmark.csv`
- `frozen/primary_manifest.json`

The helper script creates these as a new immutable evaluation materialization without
overwriting the amended artifacts or changing the selected 60 scenario IDs/order.

For amendment rows, the existing convenience CSV may have an empty pipe-delimited
`ground_truth` field while the frozen per-objective majority-vote yes/no columns are
present. The materialization deterministically reconstructs the pipe-delimited set from
those frozen columns and checks it against `k_prime`. No new annotation or model output
is introduced.

## Included paraphrase design

- 10 families
- 5 paraphrases per family
- 50 candidate variants
- 10 distinct human-confirmed ground-truth sets
- generator: GPT-5.6 Sol, external to the evaluated llama3:8b selectors
- no literal Accuracy/Runtime/Energy/CO2 tokens in the paraphrase text
- two independent human reviewers are still mandatory

The two reviewers must receive:
- their own `review_A.csv` or `review_B.csv`
- `review_reference.csv`
- `PARAPHRASE_REVIEW_INSTRUCTIONS.txt`

Do **not** send `base_scenarios.csv` to reviewers because it contains hidden ground truth.

The strict helper refuses to freeze paraphrases unless all 50 variants are unanimously
judged meaning-preserving and both reviewers independently recover the frozen base
objective set. If any variant fails, preserve the review as evidence, regenerate only
that variant before evaluation, re-review it independently, then retry.

## Included adversarial design

12 fresh cases:
- 4 ambiguous
- 4 contradictory
- 4 out_of_scope

Expected statuses are frozen before evaluation. No Phase-12-v1 adversarial text is copied.

## Workflow

From repository root:

```powershell
python .\scripts\phase12_v2_finalize_remaining_inputs.py status
python .\scripts\phase12_v2_finalize_remaining_inputs.py activate-primary
python .\scripts\phase12_v2_finalize_remaining_inputs.py prepare-inputs
```

`prepare-inputs`:
- verifies/synchronizes the 10 paraphrase bases against the frozen primary;
- regenerates deterministic review A/B packets;
- creates reviewer reference;
- freezes the 12-case adversarial set immediately.

Then obtain the two independent human reviews.

After replacing the blank `review_A.csv` and `review_B.csv` with completed files:

```powershell
python .\scripts\phase12_v2_finalize_remaining_inputs.py validate-paraphrase-reviews
python .\scripts\phase12_v2_finalize_remaining_inputs.py freeze-paraphrases
python .\scripts\phase12_v2_finalize_remaining_inputs.py validate-evaluation-freeze
```

Only when `validate-evaluation-freeze` passes may confirmatory inference begin.

## Confirmatory run, after all inputs are frozen

Verify runtime first:

```powershell
ollama --version
ollama list
```

The protocol requires the Phase-11R confirmatory runtime lock (Ollama 0.34.0 and the
frozen llama3:8b digest).

Run both selectors on the exact same frozen inputs:

```powershell
python .\scripts\phase12_v2.py run-primary --selector both
python .\scripts\phase12_v2.py run-paraphrases --selector both
python .\scripts\phase12_v2.py run-adversarial --selector both

python .\scripts\phase12_v2.py score-primary
python .\scripts\phase12_v2.py score-paraphrases
python .\scripts\phase12_v2.py score-adversarial

python .\scripts\phase12_v2.py finalize
python .\scripts\phase12_v2.py audit
```

Do not rerun if an immutable confirmatory output already exists.

## Research integrity

Do not:
- modify the frozen old primary/amendment manifests;
- modify human labels to make reviews pass;
- run V2/V3.2 before paraphrase and adversarial freezes;
- use private generator intent as ground truth;
- use model outputs to select or replace benchmark cases;
- use the evaluated llama3:8b model to generate paraphrases/adversarial cases.
