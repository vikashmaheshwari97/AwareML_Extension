# Phase 2 — Streaming Core & Responsible-AI Instrumentation

Phase 2 standardizes the measurement contract used by all five AwareML frameworks before HPC-scale meta-data generation.

## Protocol

Every framework uses test-then-train prequential evaluation. A prediction is obtained before the current example is learned. By default, a missing warm-up prediction counts as incorrect; the policy is recorded in the execution manifest.

## Performance

Run-level records contain cumulative prequential Accuracy and Macro-F1, throughput, mean prediction latency and p95 prediction latency. Window records additionally contain rolling Accuracy and rolling Macro-F1.

## Drift

ADWIN events are recorded with the current rolling-performance context. A drift episode is considered recovered when rolling Accuracy returns to within the configured tolerance (default 0.02) of its pre-drift baseline. Unrecovered episodes remain explicit. Run summaries report recovery rate, mean recovery samples and maximum observed Accuracy drop.

## Fairness

Sliding-window fairness contains Demographic Parity, Equal Opportunity, Equalized Odds, Predictive Parity and Error-Rate gaps. Phase 2 adds worst-group Accuracy, worst-group Macro-F1 and per-group support/performance. Insufficient group support is represented as `None`, never zero disparity.

## Explainability

At stream windows, adapters may emit native feature importance when the underlying framework genuinely exposes it. No statistical feature proxy is relabeled as model importance. After the framework run, AwareML computes the existing repeated-permutation diagnostic on the final recent window. Its execution time is recorded as instrumentation overhead and is excluded from framework runtime/energy comparison.

True cross-window perturbation XAI requires checkpointable model states and is intentionally not fabricated in Phase 2; that extension belongs to the later explainability research phase.

## Sustainability

CodeCarbon measures the framework streaming run only. The tracker is stopped before post-hoc explainability so framework energy/CO2 comparisons do not include diagnostic XAI overhead. Missing measurements remain null with an explicit status.

## Recorded experiment layout

When `record_experiments=True`, every framework receives an independent immutable directory:

```
artifacts/meta_experiments/<experiment_id>/
  execution_manifest.json
  run.json
  windows.jsonl
  drift_events.jsonl          # present when drift is detected
  fairness.jsonl
  explainability.jsonl
  sustainability.jsonl
```

This layout is safe for future Slurm job arrays because independent workers do not append to a shared meta-log file.

## Local validation

```powershell
python scripts\validate_phase2_instrumentation.py
python -m pytest tests -v
```

A recorded CSV run can be launched with:

```powershell
python scripts\run_recorded_benchmark.py data.csv `
  --dataset-id my_stream `
  --target target `
  --sensitive sex `
  --window 500 `
  --max-samples 5000 `
  --time-budget 60 `
  --track-sustainability
```
