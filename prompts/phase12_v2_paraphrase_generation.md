# Phase-12-v2 fresh paraphrase generation

Run this only after `python scripts/phase12_v2.py prepare-paraphrases` has selected 10 bases from the frozen 60-case benchmark.

Use an outside stronger LLM, not `llama3:8b`. For each base scenario generate exactly five natural paraphrases that preserve the complete intended meaning while varying syntax, wording, tone, and information order.

Constraints:
- Do not explicitly introduce the literal frozen labels `Accuracy`, `Runtime`, `Energy`, `CO2`/`CO₂`.
- Do not add or remove a requirement.
- Do not make an implicit requirement artificially explicit just to help the evaluated model.
- Do not reuse any legacy Phase-12-v1 paraphrase.
- Record generator model and batch for each variant.

Populate `data/journal/objective_selection_benchmark_v2/paraphrases/paraphrase_candidates.csv`. Then run `prepare-paraphrase-reviews`. Two independent human reviewers must confirm meaning preservation and independently label all four objectives before variants are frozen for evaluation.
