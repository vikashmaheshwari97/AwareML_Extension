# Phase 12 Human Annotation Guide

## What you are labeling

For each sentence, select every primary system objective that the sentence
**implies**, even when the objective name itself is not written.

Use only these four labels:

- **Accuracy** — dependable/correct predictive decisions.
- **Runtime** — rapid response, low delay, time-critical operation.
- **Energy** — battery life, electrical/power constraints, edge-device resource use.
- **CO2** — environmental/climate impact of computation.

## Important boundaries

Do not label:
- fairness,
- explainability,
- concept drift,
- usability,
- cost/money,
- model size,

unless the sentence also independently implies one or more of the four primary
objectives above.

## Ambiguity

Mark `ambiguous=yes` when a reasonable annotator cannot infer a defensible
objective subset without guessing.

Do not force a label simply because the sentence sounds generally positive.

## Independence

Complete your file independently. Do not:
- inspect another annotator's file;
- inspect `candidate_generation_intent.PRIVATE.csv`;
- ask the evaluated LLaMA for help.

Use an anonymous annotator code, not a personal name.
