# Objective Selection V3.3 — Interactive Development Upgrade

## Scientific role

V3.3 is an **interactive development successor**, not a new confirmatory benchmark result.

The frozen Phase-12-v2 experiment remains:

- **V2 fresh replay** — primary confirmatory evidence.
- **V3.2** — fresh confirmatory candidate evaluated on the same frozen benchmark.
- **V3.1** — development/post-hoc evidence only.
- **V3.3** — interactive development method introduced after observing the Phase-12-v2 result.

Do not rerun V3.3 on the same Phase-12-v2 benchmark and describe that result as independent confirmation.

## V3.3 design

V3.3 combines two ideas:

1. **V3.2 strict validation for LLM-selected objectives**
   - exact scenario-local quote;
   - V3.2 objective concept guard;
   - V3.2 negative-requirement rejection.

2. **V3.1 controlled recall recovery**
   - strong objective-specific scenario cues may recover an objective omitted by LLaMA;
   - strong cues may also recover from malformed model JSON;
   - recovery is always visible in `fallback_used` and the selection audit;
   - human review remains mandatory.

The selector uses the Phase-11R/Phase-12-v2 common runtime lock (`llama3:8b`, same frozen model digest, Ollama 0.34.0), but V3.3 itself is not frozen confirmatory evidence.

## UI behavior

The Copilot shows the finalized Phase-12-v2 V2 replay as **Primary confirmatory evidence**.

Old V3/V3.1 post-hoc benchmark panels are removed from the active Copilot evidence area. V3.1 remains documented as development/post-hoc history.

When the Copilot LLM toggle is ON, interactive objective selection uses V3.3.

When the toggle is OFF, the existing deterministic V2-style path remains available.

The ML Recommender V2 remains a separate downstream component for dataset-aware framework ranking.
