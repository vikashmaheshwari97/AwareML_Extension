# Architecture

AwareML Extension uses a **ports-and-adapters** layout. Each AutoML framework implements the same online contract (`predict_one`, `learn_one`, `reset`, metadata), while evaluation, fairness, explainability, sustainability, ranking, LLM grounding, and user studies are framework-independent services.

```text
Stream / CSV
   │
   ├─ StreamingEncoder
   │
   ├─ Framework adapters
   │    ├─ AutoStreamML
   │    ├─ AutoClass
   │    ├─ EvoAutoML
   │    ├─ OAML
   │    └─ ChaCha
   │
   ├─ Prequential runner ── drift / fairness / uncertainty
   │
   ├─ Explainability + sustainability
   │
   ├─ Objective utility + epsilon-Pareto ranking
   │
   └─ UI / grounded chat / research studies
```

The architecture is intentionally different from the original monolithic `backend_stream.py` and `forntend.py`: new frameworks can be registered by adding one adapter and one registry entry, without changing the UI or evaluation engine.

## Versioned research evidence layer

Phase 1 adds a separate experiment-data layer between framework execution and the recommender:

```text
Framework runner
      │
      ├── run summary
      ├── window metrics
      ├── drift events
      ├── fairness snapshots
      ├── explainability snapshots
      └── sustainability snapshots
              │
              ▼
artifacts/meta_experiments/<experiment_id>/
              │
        validator / reducer
              │
              ▼
versioned recommender snapshot
              │
              ├── LLM-Assisted Recommender grounding
              └── ML Recommender training/evaluation
```

The legacy 47-dataset evidence remains immutable. New dataset additions always create a new manifest/snapshot version rather than mutating V1.
