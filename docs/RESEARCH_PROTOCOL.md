# Research protocol for the AwareML Extension study

## 1. Technical evaluation

Use prequential **test-then-train** evaluation. Preserve stream order. The same sample cap, seed policy, window size and wall-clock budget must be used for all frameworks in one comparison. Report per-framework accuracy/F1 trajectories, drift events, runtime, fairness where applicable, and measured sustainability values with uncertainty.

Primary evaluation should span multiple real temporally ordered/native streams, not only Adult treated as a stream. Synthetic drift generators are diagnostic supplements rather than substitutes for real streams.

## 2. Meta-recommender evaluation

Meta-learning rows from the same dataset must never be split across train/test folds. Use leave-one-dataset-out or GroupKFold. Compare against simple and strong baselines. Recommended reporting: Top-1 selection accuracy, normalized regret, rank correlation, per-dataset results, and bootstrap confidence intervals. If confidence estimates are shown in the UI, evaluate their calibration separately.

## 3. Objective utility and Pareto analysis

Metrics are robustly mapped to [0,1], with lower-is-better objectives inverted. User weights are normalized. If a metric was not measured (for example energy when CodeCarbon is disabled), it is excluded and the available weights are renormalized for that candidate. “Near-Pareto” means epsilon-nondominated in the available normalized objective space; epsilon is displayed to the user.

Energy and CO2 are not automatically independent objectives. When carbon intensity is fixed, CO2 can be a near-deterministic transformation of energy. The Decision Lab therefore displays their correlation and warns against accidental double-weighting.

## 4. Sustainability

Run one warm-up followed by at least five measured repetitions when producing paper claims. Record machine, CPU/GPU, RAM, OS, Python and CodeCarbon versions, region/carbon-intensity context, start time, sample count and time budget. Report mean, standard deviation, median and bootstrap 95% CI. If measurement fails, store null and status; never replace missing energy with zero.

## 5. Fairness

Sensitive attributes require explicit analyst confirmation. Report group support with each fairness result. Metrics that cannot be estimated because of insufficient group/positive/negative support remain undefined. Do not turn a single aggregate fairness score into a claim of fairness.

## 6. Explainability

Use model-native importance only when genuinely available; otherwise use repeated permutation importance. Report stability, consistency, sensitivity, sparsity and deletion fidelity as diagnostics of the explanation procedure, not causal guarantees. Avoid random fallback importance.

## 7. Trust calibration experiment

The operational recommendation and the experimental manipulation are isolated. Participants receive the same explanation template/length/tone while recommendation reliability varies: correct (top observed utility), weak (second-best), wrong (lowest observed utility). Measure trust, perceived correctness, acceptance, decision confidence and response time. Primary outcomes include wrong-acceptance rate and discrimination in trust between reliable and unreliable recommendations.

## 8. Information-seeking study

Use interaction logs plus think-aloud or semi-structured interviews. Track whether the first answer is accepted, follow-up depth and categories (evidence request, explanation probe, challenge, comparison/counterfactual, clarification). The goal is to study when users actively seek evidence versus accepting fluent explanations.
