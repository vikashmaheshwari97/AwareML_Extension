# Phase 14 — Sustainability measurement protocol

## Measurement boundary

Framework runtime and CodeCarbon measurement stop before post-hoc
explainability so XAI computation does not contaminate framework
Runtime/Energy/CO₂ comparisons.

## Recorded context

Every measured run records, where available:

- CPU model;
- physical and logical CPU counts;
- GPU inventory;
- RAM;
- Python version;
- country ISO;
- region;
- CodeCarbon version;
- measurement backend;
- carbon intensity in gCO₂/kWh;
- measurement duration;
- warm-up seconds / samples;
- repetition ID and planned repetition count;
- measurement status;
- failure reason.

Missing values remain missing and are never replaced with zero.

## Carbon intensity

Carbon intensity is derived unit-safely from the measured pair:

`(CO2_kg × 1000) / energy_kWh`

when both quantities are available.

## Repeatability

The formal paper-ready Phase-14 sustainability gate requires **at least five
independent measured repetitions per framework**.

The default runner uses seeds:

`42, 43, 44, 45, 46`

For every framework report:

- Runtime mean ± sample SD;
- Energy mean ± sample SD;
- CO₂ mean ± sample SD.

Median is also retained in the repeatability table. Sample SD uses `ddof=1`.

Runs with fewer than five repetitions are retained as development/exploratory
evidence but do not satisfy the paper-ready gate.

## Dataset-specific evidence

Repeatability evidence is indexed by the exact combination of:

- canonical dataframe SHA256;
- target;
- sensitive attribute;
- positive label.

Each invocation receives a timestamped run directory, so a new dataset,
configuration, or rerun never overwrites previous evidence.

The registry is stored under:

`artifacts/phase14/repeatability/registry.json`

## Failure handling

Statuses remain explicit:

- `measured`
- `measurement_incomplete`
- `measurement_failed`
- `not_measured`

A failed or unavailable Energy/CO₂ measurement remains missing.
