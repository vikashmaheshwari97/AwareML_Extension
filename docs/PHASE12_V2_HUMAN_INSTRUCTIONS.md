# Human-data instructions for Phase 12 v2

## Realism filter
The realism reviewer must be uninvolved in AwareML development and must not see private generation intent or any selector outputs. For every candidate, mark `keep=yes/no`, `ambiguous_or_unclear=yes/no`, reviewer ID, and optional notes. A candidate cannot be both kept and marked ambiguous/unclear.

Prefer statements that sound like something a real user would enter into an AutoML assistant. Drop repetitive, synthetic-sounding, checklist-like, or unnaturally clean text. This is a realism filter, **not** a model-performance filter.

## Human-written pool
Collect at least 30 genuinely human-written candidate goals from people not involved in AwareML. Ask: “Imagine you are deploying an automated ML system. In your own words, what would you tell the tool you need from the system?” Do not teach them the four labels first. The primary scenario text must not literally name the frozen labels.

## Independent ground-truth annotation
Use exactly three distinct annotators who are not Vikash or Radwa and who have not seen selector outputs or private generation intent. They work independently before any discussion.

For each scenario mark yes/no for:
- Accuracy — predictive correctness/quality, error avoidance, dependable predictions/decisions
- Runtime — speed, latency, delay, deadline, time-to-response
- Energy — battery, charging, electrical power/electricity use
- CO2 — carbon/emissions/climate/environmental footprint

Also mark `ambiguous=yes` when the scenario is genuinely unclear. Do not infer an objective merely because it is generally desirable.

Ground truth is per-objective 2-of-3 majority vote. Generator intent is never used. The code separately flags hard cases when there is a majority ambiguity vote, low pairwise set agreement, multiple 2-to-1 objective splits, or a zero-label majority. Hard cases are excluded from the primary 60 and retained for separate reporting.

## Paraphrase review
Two independent reviewers must review all 50 fresh paraphrases. Each reviewer marks whether meaning is preserved and independently marks the four objectives. A variant enters the scored paraphrase set only if both reviewers confirm preservation and both recover the base scenario's independent human ground-truth objective set.

## Adversarial set
Write 10–15 fresh cases that were not used in V3.1 development. Pre-register one expected status before running any selector: `ambiguous`, `contradictory`, or `out_of_scope`. Include vague requests, explicit same-objective contradictions, and unsupported goals. Do not inspect model output while writing/revising these cases.
