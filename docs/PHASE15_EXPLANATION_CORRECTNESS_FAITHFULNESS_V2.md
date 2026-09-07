# Phase 15 — Explanation Correctness & Faithfulness V2

## Purpose

Phase 15 keeps two evaluation questions separate.

**Correctness** asks whether factual statements in an explanation agree with the
structured evidence supplied to the explanation system.

**Faithfulness** asks whether the explanation responds appropriately when
decision-relevant evidence changes.

A correct explanation can still be unfaithful if it ignores a counterfactual
change. A faithful explanation can still be incorrect if it consistently
tracks the wrong evidence. AwareML therefore reports the two dimensions
separately.

## Explanation sources

The Phase-15 verifier supports the four roadmap sources:

1. Stage B — setup/recommendation explanation
2. Stage E — fairness explanation
3. Stage F — SHAP/XAI explanation
4. Stage F — conversational answer

## General evidence verifier

The deterministic verifier extracts and checks claims about:

- Accuracy
- Runtime
- Energy
- CO2
- Statistical/Demographic Parity (SPD/DP)
- Equal Opportunity (EO)
- Equalized Odds (EOdds)
- Group Brier-score gap
- Group Expected Calibration Error (ECE) gap
- SHAP values
- feature ranking
- framework ranking

The verifier does not use a second LLM judge. This avoids conflating evaluator
model behavior with explanation-model behavior.

### Correctness metrics

- **claim precision** = supported extracted factual claims / all extracted
  factual claims
- **supported-claim rate** = supported claims / verifiable claims
- **numeric correctness** = numerically correct claims / numerically verifiable
  claims
- **citation validity** = cited evidence keys that exist / all cited keys
- **citation coverage** = claims carrying at least one evidence citation /
  extracted claims
- **decision consistency** = correct ranking/decision claims / verifiable
  ranking/decision claims
- **unsupported-claim rate** = unsupported claims / extracted claims
- **contradiction rate** = claims contradicted by evidence / extracted claims

Unavailable denominators remain N/A rather than being converted to zero.

## Faithfulness V2

Faithfulness V2 uses external evidence interventions inspired by the
counterfactual-explanation testing pattern in FaithLM:

https://github.com/ynchuang/FaithLM

The AwareML adaptation does **not** claim access to hidden model states,
attention weights, or internal attribution. Instead, it changes the structured
evidence that the explanation generator receives, regenerates the explanation,
and checks whether the changed evidence is acknowledged.

Relevant intervention metrics include:

- changed-evidence acknowledgement
- numeric update accuracy
- stale-claim rate
- decision-update consistency when a ranking/feature decision changes
- explanation change
- a separate irrelevant-control invariance score

Correctness scores are not folded into the faithfulness score.

## Controlled benchmark

The frozen controlled benchmark contains six independent synthetic evidence
contexts and all four explanation sources, giving 24 base explanation cases.

Each base case produces:

- one known-correct explanation
- one known-incorrect explanation

Therefore the Track-2 stimulus bank contains:

- 24 known-correct stimuli
- 24 known-incorrect stimuli
- 48 total stimuli

The controlled frozen artifacts validate the verifier and intervention
methodology. They are **not** an estimate of any particular LLM's empirical
correctness or faithfulness.

## Empirical model run

Use:

```powershell
python -m scripts.run_phase15_llm_benchmark
```

This runs the configured local Ollama model on the controlled evidence cases and
stores timestamped empirical results under:

`artifacts/phase15/llm_runs/`

Model-facing results are intentionally not auto-frozen. They must be reviewed
before being used for journal-facing LLM performance claims.

## Live Dataset Probe

The Advanced Research Labs UI includes a Live Dataset Probe. It uses the active
AwareML dataset/recommendation/run evidence, but is explicitly labeled:

**Exploratory — NOT frozen journal evidence.**

It never replaces the controlled benchmark.

## Track 2 trust stimulus bank

The frozen researcher bank contains ground-truth labels and verifier results.

The participant-facing bank excludes:

- correctness labels
- error types
- verifier metrics

This prevents accidental disclosure of the intended condition to study
participants.

## Phase-15 completion gate

The controlled implementation gate requires frozen:

- `explanation_correctness_v1`
- `faithfulness_v2`
- `trust_stimulus_bank_v1`

Each frozen artifact contains a SHA256 manifest.

A separate empirical LLM run is required for any claim about a specific LLM's
actual explanation correctness or faithfulness.
