AwareML UI Label Test Fix
===========================

Purpose
-------
The professional UI polish intentionally changed the visible workspace label from:

  Explanation Integrity · Phase 15

to:

  Explanation Integrity

The implementation is correct, but tests/test_phase15_ui.py still asserted the
old display string. This patch updates only that stale UI-label expectation.

Scientific / study logic changed: NO
Phase-15 evaluator logic changed: NO
Phase-16 trust-calibration logic changed: NO
Stimuli / randomization / storage / analysis changed: NO

Apply
-----
From the AwareML_Extension repository root:

  python .\APPLY_UI_LABEL_TEST_FIX.py

Then:

  pytest -q tests/test_phase15_ui.py
  pytest -q
