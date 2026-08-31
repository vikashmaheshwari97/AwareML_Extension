# Phase 10 — Journal Protocol Lock & Roadmap Realignment

## Purpose

Phase 10 does **not** add another model, recommender, UI feature, fairness
metric, or faithfulness metric. Its purpose is to stop the journal extension
from drifting methodologically before the reviewer-driven experiments begin.

The completion artifact is:

`data/journal/protocol_v1/journal_experimental_protocol_v1.json`

together with its SHA256 checksum and active-protocol marker.

## 10.1 Freeze the current engineering baseline

The Phase-10 baseline is the current `main` engineering state immediately
before journal-specific realignment:

- repository: `vikashmaheshwari97/AwareML_Extension`
- branch: `main`
- commit: `bee541f2914f55db099a99693b3b6fae50a89996`
- CPython: exactly `3.8.10`

The protocol hashes the principal frozen artifacts from Phases 5–9, including:

- canonical 705-run Meta-Dataset V2;
- 235-row Recommender V2 training table;
- Recommender V2 model manifest;
- Copilot v1 manifest;
- Faithfulness v1 manifest;
- Research UI v2 manifest;
- dataset split manifests;
- reproducible requirements.

No Phase-10 script retrains models or changes the previous phase artifacts.

## 10.2 Resolve the roadmap ambiguity

The roadmap contains the phrase **“23 meta-training datasets.”**

That wording is inconsistent with both the accepted CIKM paper and the current
repository manifests:

- `train_v1_47.yaml`: `purpose=meta_train`, `expected_count=47`, frozen;
- `test_v1_23.yaml`: `purpose=heldout_test`, `expected_count=23`.

Therefore the Journal Experimental Protocol adopts the canonical terminology:

- **47 development/meta-training datasets**
- **23 final held-out evaluation datasets**

There is a second practical issue: the current repository's
`test_v1_23.yaml` does not contain the 23 authoritative dataset IDs and is not
frozen. Phase 10 deliberately does **not invent them**.

Operational rule:

> Phases 10–17 may not load or use the final 23 held-out datasets. Phase 18 is
> blocked until the identities are recovered from authoritative project
> history, populated, checked for zero overlap with the 47 development
> datasets, and frozen.

This also resolves the roadmap's ambiguous “1.6/2.1” reference conservatively:
it is **not permission to consume the final held-out set during explanation
development**.

## 10.3 Freeze the objective vocabulary

The journal NL objective-selection benchmark has exactly four labels:

1. **Accuracy**
2. **Runtime**
3. **Energy**
4. **CO2**

These correspond to the existing internal keys:

- `accuracy`
- `runtime`
- `energy`
- `co2`

Fairness and explainability are HCAI requirements/evidence dimensions and are
not labels in this four-objective benchmark.

The primary Phase-12 task is **multi-label objective-set inference**. It does
not evaluate continuous objective weights.

Downstream weighting remains a separate Phase-11 decision and must be frozen
before Phase-12 execution.

## 10.4 Freeze the journal LLM

The roadmap and accepted paper specify **LLaMA 3 8B via Ollama**.

The journal protocol therefore uses:

- provider: Ollama
- exact tag: `llama3:8b`
- silent fallback: **forbidden**
- JSON output
- temperature: `0.0`
- top-p: `1.0`
- seed: `42`
- `num_predict`: `512`

At Phase-10 freeze time AwareML also records:

- exact Ollama version;
- exact model digest;
- model details exposed by Ollama;
- SHA256 of the prompt template;
- SHA256 of the JSON schema.

If only `llama3:latest`, `llama3.2:3b`, or another model is installed, the
freeze **fails**. The journal benchmark must not silently switch models.

The current Phase-7 Copilot code is not modified in Phase 10. Phase 11 will
wire its roadmap-aligned objective-selection V2 to the frozen journal
prompt/schema/model contract.

## Completion gate

Phase 10 is complete only when:

1. preflight validation passes;
2. exact CPython 3.8.10 is active;
3. engineering baseline commit/artifact hashes match;
4. objective vocabulary and schema match the current four-objective code;
5. held-out policy is locked;
6. exact `llama3:8b` is installed;
7. Ollama version and model digest are captured;
8. protocol JSON is created;
9. protocol SHA256 validates;
10. final validation passes.

## What Phase 10 intentionally does not do

- does not modify the 705 meta-logs;
- does not retrain Recommender V2;
- does not modify current Copilot behavior;
- does not modify Faithfulness;
- does not modify the Research UI;
- does not access the 23 held-out dataset contents;
- does not choose the final downstream weighting policy;
- does not redefine near-Pareto beyond recording the current epsilon-Pareto engineering state.

Those are later roadmap phases.
