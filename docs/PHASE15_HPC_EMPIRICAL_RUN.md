# Phase 15 — Final 24-case empirical LLaMA-3 evaluation on UTHPC Rocket

## Scientific purpose

This HPC campaign is the **actual model-performance evaluation** for Phase 15.

It is separate from:

- the frozen controlled methodology benchmark;
- the exploratory Live Dataset Probe;
- the Track-2 trust-calibration stimulus bank.

The campaign evaluates all 24 controlled Phase-15 cases:

- 6 controlled dataset contexts;
- 4 explanation sources per context:
  - Stage B recommendation;
  - Stage E fairness;
  - Stage F SHAP/XAI;
  - Stage F conversational answer;
- one original explanation + one relevant counterfactual explanation per case.

## Frozen runtime

Model:

`llama3:8b`

Expected model digest:

`365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1`

Ollama:

`0.32.14`

Prompt:

`phase15_explanation_prompt_v4`

Generation:

- temperature = 0.0
- top_p = 1.0
- seed = 42
- num_predict = 512
- stream = false

If model digest or Ollama version differs, the final job refuses to run.

## Why a Slurm array

Each case is independent and becomes one array task.

Default:

`0-23%4`

So at most four cases run at the same time.

Every task gets one GPU and runs its own job-local Ollama server. The shared
model files live under `$HOME/.ollama/models`.

## Output preservation

Each case has append-only attempts:

`artifacts/phase15/hpc_runs/<campaign>/cases/task_###__<case>/attempts/<attempt>/`

A failed retry never overwrites an earlier attempt.

Only a fully complete case receives:

`SUCCESS.json`

Resume logic skips valid successful cases and resubmits only missing/failed task
IDs.

## UTHPC usage

Do not run inference on the login node.

Run setup and inference via `sbatch`.

### 1. Pull latest code

```bash
cd ~/AwareMLExtension
git pull origin main
git status
```

If your directory is `~/AwareML_Extension`, the scripts detect that name too.

### 2. Validate locally on the cluster filesystem

These checks are lightweight and do not perform LLM inference:

```bash
$HOME/.conda/envs/awareml-main/bin/python -m scripts.validate_phase15_hpc_runner
```

### 3. Prepare pinned Ollama + model

```bash
sbatch hpc/production/phase15/setup_phase15_ollama.sbatch
```

Monitor:

```bash
squeue -u $USER
```

Inspect the corresponding Slurm output and:

```bash
ls artifacts/phase15/hpc_setup
```

The setup job must pass the exact Ollama version and model-digest lock.

### 4. Submit final 24-case campaign

Default A100-40G, four concurrent cases:

```bash
bash hpc/production/phase15/submit_phase15_campaign.sh \
  --campaign-id phase15_llama3_8b_final_v1 \
  --concurrency 4 \
  --gpu-gres gpu:a100-40g:1
```

If A100 queue pressure is high, reduce concurrency without changing the
scientific protocol:

```bash
--concurrency 2
```

Do not mix GPU types within the same final campaign unless this is explicitly
documented and accepted before examining results.

### 5. Monitor

```bash
squeue -u $USER
```

After completion:

```bash
sacct -j <ARRAY_JOB_ID> \
  --format=JobID,State,Elapsed,NodeList,AllocTRES%50
```

### 6. Inspect campaign status

```bash
python hpc/production/phase15/resume_phase15_campaign.py \
  --campaign-root artifacts/phase15/hpc_runs/phase15_llama3_8b_final_v1
```

### 7. Resume only failed/missing cases

```bash
bash hpc/production/phase15/resume_phase15_campaign.sh \
  artifacts/phase15/hpc_runs/phase15_llama3_8b_final_v1 \
  4
```

Successful cases are not repeated.

### 8. Reduce results

Only after all 24 cases have valid success markers:

```bash
python hpc/reduce/phase15/collect_phase15_results.py \
  --campaign-root artifacts/phase15/hpc_runs/phase15_llama3_8b_final_v1
```

The reducer writes a timestamped directory containing:

- `phase15_empirical_summary.json`
- `phase15_empirical_full.json`
- `correctness_by_case.csv`
- `faithfulness_by_case.csv`
- `summary_by_source.csv`
- `failures.json`
- `collection_manifest.json`

These results are **not auto-frozen**. Review them before making a
journal-facing model-performance claim.
