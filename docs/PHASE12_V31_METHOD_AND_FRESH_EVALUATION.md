# Phase-12 V3.1 method and fresh-evaluation plan

## Why V3.1 exists

Frozen V1 had high recall and low precision (systematic over-selection). V3 fixed over-selection but became too conservative, reducing recall and paraphrase stability. V3.1 therefore uses a hybrid rule:

1. exact locked LLaMA proposes per-objective decisions;
2. scenario-local semantic evidence independently checks each objective;
3. unsupported LLaMA additions are rejected;
4. strong explicit semantic evidence can recover an LLaMA omission;
5. broad generic phrases do not count by themselves;
6. all recovery is shown to the user and remains human-reviewable.

## Important boundary

V3.1 was designed after inspecting the frozen Phase-12 v1 and V3 behavior. Therefore the 55 legacy cases, 10 legacy paraphrase families and 14 legacy adversarial cases are **development diagnostics only** for V3.1.

## Fresh evaluation design

The fresh test should be collected only after the V3.1 method is frozen:

- 60 primary scenarios;
- 15 each for k′=1,2,3,4;
- at least 24 genuinely human-written scenarios (>=40%);
- 3 independent objective annotators;
- >=2 independent paraphrase semantic reviewers, preferably blinded per-variant objective annotation;
- a new adversarial set unseen during V3.1 development;
- private generator-intent metadata unavailable to annotators and selector code until evaluation ends.

Do not resurrect realism-rejected legacy k′=4 checklist-like cases merely to fill the bucket. Collect natural multi-objective scenarios instead.
