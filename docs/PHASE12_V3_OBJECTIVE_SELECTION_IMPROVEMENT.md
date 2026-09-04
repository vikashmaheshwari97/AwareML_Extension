# Objective Selection V3: Evidence-Grounded Conservative Selector

Phase-12 v1 remains frozen and unchanged. Its main failure pattern was high recall with lower precision and systematic over-selection. V3 is an improvement candidate designed specifically around that observed failure mode.

## Method

V3 keeps the exact locked journal LLaMA runtime but changes the interactive selection protocol:

1. preflight handles empty, generic, clearly out-of-scope and explicit contradiction cases;
2. LLaMA returns one decision for each frozen objective;
3. each selected objective must provide scenario-local evidence;
4. a transparent objective-specific semantic guardrail must support the selection;
5. low-confidence selections are rejected;
6. downstream weights remain `equal_selected_v1`;
7. human review remains mandatory.

## Research interpretation

V3 diagnostics on the original 55 cases are post-hoc development evidence because V3 was motivated by the v1 failure analysis. They must not replace the frozen v1 result in the paper. A new final claim requires a fresh independently annotated benchmark after the V3 method is fixed.
