# Phase 3 — Framework hardening and journal-grade responsible-AI diagnostics

Phase 3 hardens the Phase-2 measurement path before large-scale HPC generation. The goal is to make every result auditable: a successful-looking number or plot must correspond to a supported measurement rather than a fallback placeholder.

## Why this phase exists

The first recorded Adult/sex Phase-2 run exposed four useful engineering issues:

1. Model-level XAI used repeated permutation only; SHAP/LIME were installed but were not part of the actual Phase-2 execution path.
2. AutoStreamML and AutoClass produced all-zero permutation vectors that were labelled `ok`. A zero vector is now treated as degenerate/unsupported rather than a valid explanation.
3. AutoStreamML drift episodes could appear to recover one sample after ADWIN even when no measurable post-drift degradation had occurred. Phase 3 requires a post-drift assessment horizon before recovery can be declared.
4. ChaCha fell back because FLAML AutoVW's `learn()` was called without the `predict()` state required by AutoVW. The native adapter now follows FLAML's predict-then-learn contract and uses FLAML `Domain` objects for the search space.

## XAI cascade

`RunConfig.xai_method="auto"` uses:

1. `SHAP/Kernel`
2. `LIME/tabular`
3. `permutation/repeated`

SHAP and LIME are attempted only when the framework provides genuine class-probability APIs. A hard-label pseudo-probability is deliberately not used. Every attempt is logged in `method_attempts`, and the selected method is logged in `method`.

For mixed tabular data, LIME receives the categorical columns learned by the streaming encoder so encoded categories are not silently treated as ordinary continuous variables.

If a method produces an all-zero or non-finite aggregate signal, it is rejected. If every requested method fails or degenerates, the explanation status is `unsupported` and no feature-importance plot should be presented as scientific evidence.

The final-model explanation is a replay on the recent window, not a reconstruction of historical predictions. Phase 3 compares replay accuracy with the latest prequential rolling accuracy and emits a warning when the gap is material.

## Fairness protocol

A protected attribute has an explicit model-access policy:

- `audit_only` (default): keep the attribute for fairness measurement, remove it from model inputs.
- `include`: the model may use the attribute and AwareML also audits it.

The choice is stored in the run configuration and exported YAML. This avoids silently changing the meaning of an Adult/sex experiment.

Temporal fairness summaries use only observed/defined windows. Missing or insufficient-support windows are not converted to zero. For each disparity metric AwareML reports the mean, maximum, p95, volatility and time-weighted mean, plus the worst observed group Accuracy and Macro-F1.

## Drift recovery protocol

An ADWIN event opens a drift episode. Recovery is not eligible until `drift_min_assessment_samples` have been observed. If rolling Accuracy never drops beyond the configured tolerance during the assessment horizon, the episode is marked `degradation_observed=false` and recovery is *not applicable* rather than "1 sample".

For episodes with real degradation, AwareML records the minimum post-drift Accuracy, Accuracy drop, recovery sample and samples-to-recover.

## Framework hardening

- AutoStreamML and AutoClass expose genuine River probability APIs for post-hoc SHAP/LIME when their active models support them.
- EvoAutoML and OAML isolated workers support batch prediction/probability requests, reducing per-row subprocess overhead during XAI.
- ChaCha calls AutoVW `predict()` before `learn()` on every sample, including one-label warm-up, and uses `flaml.tune.loguniform` domains for tunable VW parameters.
- OAML retains explicit `online` and `gama` modes. `gama_available=true` does not mean GAMA was used; provenance records whether GAMA actually fitted.

## Local validation before Phase 4/HPC

Run the Phase-3 validator and complete test suite, then perform:

1. Adult with `sex`, using the default `audit_only` fairness policy and `xai_method=auto`.
2. A drift-rich synthetic/real stream (for example high-gradual Hyperplane) without fairness to validate the revised drift-recovery protocol.
3. A short ChaCha-only run to confirm the local FLAML/Vowpal Wabbit build remains on `FLAML AutoVW / ChaCha native` rather than the transparent fallback.
4. Optional explicit `--xai-method shap` and `--xai-method lime` runs to verify both paths independently.

Large-scale Slurm generation starts only after these checks are clean.
