# AwareML Research-Grade Visualization Upgrade v5

This patch is visual/diagnostic only. It does not alter benchmark outcomes, fairness values, XAI values, Phase-16 evidence, recommender rankings, or sustainability measurements.

## Fairness Lab

- Replaces the aggregate fairness figures with a paper-style **window evidence + run aggregate** figure. Small points are actual window values; diamonds are recorded run-level values.
- Adds a fairness-definition **rank-flow** plot to expose rank instability across DP, EO, EOdds, and error-rate parity.
- Adds a **mean-vs-worst robustness frontier** rather than an area-encoded bubble chart.
- The dashboard is one active dataset, so frameworks are rows and stream windows are repeated observations. It does not fabricate a multi-dataset experiment.

## Decision Lab

- Replaces bubble contribution encoding with **exact stacked utility decomposition**.
- Replaces radar geometry with a **direction-aligned objective profile**.
- Pareto view uses fixed marker size and adds a clearly defined **2D optimal compromise**: the exact Pareto point closest to the run-relative ideal corner.
- Replaces the circular correlation network with a **signed arc diagram** plus **ranked exact Spearman pairs**.

## System-level explainability

- Replaces bubble evidence maps with **metric-wise ranked lollipop panels** using separate quantitative axes.
- Adds an XAI **rank-flow** for directional metrics; Sparsity is not assigned a universal better direction.
- No overall XAI score is invented.

## Sustainability Lab

- Adds a three-axis **measured resource scoreboard** for Runtime, Energy and CO2.
- Adds a resource-efficiency **rank-flow**.
- Adds an exact Runtime-Energy **Pareto frontier** with a transparent 2D compromise marker.
- Adds a **carbon-accounting consistency** view comparing recorded carbon-intensity metadata with observed CO2/Energy intensity.
- Raw measurement provenance remains available in a collapsed expander.

## Scientific boundary

All aggregate or derived quantities introduced by these figures are explicit, auditable transformations of already recorded data. The patch does not create new experimental observations.
