# Phase-12-v2 external scenario generation prompt

Use an outside, stronger LLM than the evaluated `llama3:8b`. Generate the five batches independently; do not show one batch to the generator while producing another.

## Fixed hidden concepts
The benchmark will later ask humans to infer four hidden deployment priorities: predictive correctness/quality, response speed/timing, electrical/device resource use, and computational/deployment environmental footprint.

**Do not use the literal frozen labels `Accuracy`, `Runtime`, `Energy`, `CO2`/`CO₂` anywhere in scenario text.** Do not use framework names, metric names, weights, or numeric objective labels.

Each statement must sound like something a real user might tell an AutoML assistant. It must describe a scenario/goal rather than list objectives. Vary domain, formality, length, outcome-first vs constraint-first framing, directness, and vocabulary. Do not repeat templates. Do not make every implication equally obvious.

Private generation intent is design metadata only. It must never be shown to human annotators, the evaluated selectors, or used as ground truth.

## Five independent batches

| Batch | Style emphasis | k'=1 | k'=2 | k'=3 | k'=4 | Total |
|---|---|---:|---:|---:|---:|---:|
| A | deployment-context / professional | 7 | 7 | 8 | 8 | 30 |
| B | constraint-first / operational | 6 | 6 | 9 | 9 | 30 |
| C | outcome-first / domain-rich | 5 | 6 | 9 | 10 | 30 |
| D | informal / mixed sentence length | 4 | 6 | 9 | 11 | 30 |
| E | complex multi-constraint / natural trade-offs | 3 | 5 | 10 | 12 | 30 |
| **Total** | | **25** | **30** | **45** | **50** | **150** |

Oversampling k'=3/4 is deliberate because the legacy benchmark produced no human-majority k'=4 cases. The final benchmark is **not** selected from generator intent; it is selected only after independent human majority labels.

For each generated case record two files separately:

`generated_candidates.csv`: `scenario_id,scenario,domain,style,generation_batch`

`private/generated_intent.PRIVATE.csv`: `scenario_id,intended_k_prime,generation_intent`

Use `|` between private labels, in canonical order, e.g. `Accuracy|Energy|CO2`. Do not copy this private file into annotation packets.
