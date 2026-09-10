AwareML Phase-17 Classifier Fix
================================

Cause
-----
The Information-Seeking validator expected:

    "What does this metric mean?" -> clarification

but the classifier checked evidence keywords before clarification keywords.
Because the sentence contains "metric", it was incorrectly categorized as
evidence_request.

Fix
---
Clarification phrases now take precedence over generic evidence words such as
"metric" and "data".

A regression test for the exact validator phrase is also added.

This does not change:
- Phase-16 Trust Calibration;
- Information-Seeking storage;
- participant flow;
- manual qualitative coding;
- analysis formulas;
- design/finalization gates.

Apply
-----
Extract into the AwareML_Extension repository root, then run:

    python .\APPLY_PHASE17_CLASSIFIER_FIX.py

Then:

    pytest -q tests/test_phase17_information_seeking.py
    python -m scripts.validate_phase17_information_seeking
    pytest -q
