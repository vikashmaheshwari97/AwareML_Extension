# Canonical near-Pareto specification

**Specification ID:** `epsilon_pareto_v1`  
**Canonical epsilon:** `0.05`  
**Normalization:** `robust_quantile_05_95_all_higher_v1`

## Mathematical definition

Let `z_i in [0,1]^m` be the normalized desirability vector for candidate `i`, after converting every objective to higher-is-better. Candidate `j` epsilon-dominates candidate `i` when:

1. `z_jk >= z_ik - epsilon` for every jointly available objective `k`; and
2. `z_jk > z_ik + epsilon` for at least one objective `k`.

Candidate `i` is **epsilon-Pareto / near-Pareto** when no other candidate epsilon-dominates it. The journal uses `epsilon = 0.05`. At `epsilon = 0` the relation reduces to ordinary Pareto nondominance.

## Canonical use

- Pre-run ML Recommender V2 / 3D Decision Space: epsilon = 0.05.
- Post-run Decision Lab: epsilon = 0.05 for journal reporting.
- Paper: report this definition and epsilon exactly.
- Sensitivity views may vary epsilon, but must be labelled as sensitivity analysis.
