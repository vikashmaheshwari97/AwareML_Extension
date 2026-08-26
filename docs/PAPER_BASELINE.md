# Accepted-paper baseline

This file records the claims and scope of the uploaded accepted CIKM paper **“AwareML: Transparent, Controllable, and Sustainable Streaming AutoML”** so the extension does not accidentally rewrite the original contribution while evolving the system.

## Original system scope

The accepted paper presents AwareML as one interface over five streaming AutoML frameworks: **AutoStreamML, AutoClass, EvoAutoML, OAML, and ChaCha**. The workflow is organized into six stages: context setup; pre-execution recommendation; temporal performance/drift comparison; outcome analysis/ranking; fairness and interpretability; and LLM-assisted explanation.

The paper’s meta-learning recommender is described as being constructed from a meta-log covering **47 real-world and synthetic stream datasets** and evaluated on **23 held-out datasets**. The evaluation uses **100 preference profiles per held-out dataset (2,300 cases)** and reports **84.04% Top-1 selection accuracy** and **0.097 normalized regret**. The text describes **XGBoost** as the default supervised predictor for framework-metric utility prediction.

The paper evaluates accuracy, runtime, energy and CO2 as recommendation objectives. It uses CodeCarbon for sustainability tracking; SHAP with LIME/permutation fallback for explanations; and a local LLaMA-family model through Ollama for natural-language explanation.

The usability study contains **25 participants** and reports **80% average task completion** and **73.3%** of participants indicating increased confidence from LLM explanations.

## Scope explicitly acknowledged as limited

The demonstration centers on **Adult treated as a simulated stream**, and the conclusion calls for evaluation on broader native streaming datasets and a larger user study. The extension treats this as the primary experimental gap rather than presenting the accepted demo as sufficient evidence for general streaming behavior.

## Extension rule

New experiments must be reported as **extension results**, not retroactively attributed to the accepted paper. Likewise, changes in recommender model family, fairness definition, explanation diagnostics, or sustainability protocol must be documented as methodological changes.
