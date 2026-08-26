# AwareML Phase 9 User Manual

## 1. What Phase 9 is

Phase 9 is the **Research UI V2** for AwareML. It integrates the validated backend work from:

- **Phase 6** — ML Recommender V2
- **Phase 7** — LLM Copilot with mandatory human review
- **Phase 8** — Faithfulness / AEF layer

The signature visualization is the **interactive 3D multi-objective Decision Space**.

---

## 2. Recommended demo path for your supervisor

Use the **Synthetic drift** stream first because it is fast, easy to explain, and already aligned with the UI screenshots.

### Recommended settings

- **Source:** `Synthetic drift`
- **Synthetic samples:** `6000`
- **Target:** `target`
- **Sensitive attribute:** `group`
- **Protected attribute usage:** `Audit only` for the first demo
- **Positive label:** `1`

Why this setup is good:
- it creates a simple classification stream,
- it has a clear sensitive attribute for fairness analysis,
- it is small enough for a clean live demo,
- it produces drift-related behavior that is visible in the observatory.

If you want a custom dataset, upload a **classification CSV** with:
- one clear **target column**,
- an optional **sensitive attribute** column,
- preferably at least **500 rows**,
- no extremely wide schema for the first demo.

---

## 3. Installation

1. Extract the ZIP into your PyCharm project root.
2. Run:

```powershell
python -c "from awareml.ui_v2.pages import PAGE_REGISTRY_V2; print('PHASE9 IMPORT OK', len(PAGE_REGISTRY_V2))"
```

Expected output:

```text
PHASE9 IMPORT OK 9
```

3. Run tests:

```powershell
pytest .\tests\test_phase9_ui.py -q
```

4. Validate Phase 9:

```powershell
python .\scripts\validate_phase9_ui.py
```

5. Freeze / complete Phase 9:

```powershell
python .\scripts\complete_phase9.py
```

6. Launch the app:

```powershell
python -m streamlit run app.py
```

---

## 4. Theme and appearance

The sidebar now has an **Appearance** selector:
- **System**
- **Dark**
- **Light**

Use **Dark** for projector / demo mode. Use **Light** if your supervisor prefers printed-paper style visuals.

---

## 5. What each page does

### A. Command Center
Use this as the **opening page**.

Explain:
- this is the integrated Research OS,
- the dataset context is shared across all pages,
- the backend status of Phase 6, 7, and 8 is shown,
- the 3D Decision Space preview appears from pre-run recommendation evidence.

### B. Run Studio
Use this page to:
- load the stream,
- select target and sensitive attribute,
- configure the shared protocol,
- execute the benchmark.

This page is the main operational starting point.

### C. 3D Decision Space
Use this page to show **pre-run framework selection**.

Explain the axes clearly:
- **X = Accuracy**
- **Y = Runtime**
- **Z = Energy**
- **Marker size = CO2**

Also explain:
- the sliders change preference weights,
- this reranks the same predicted framework outcomes,
- it does **not** rerun AutoML.

### D. Streaming Observatory
Use this after a benchmark run.

Show:
- prequential accuracy,
- macro-F1,
- rolling accuracy,
- prediction latency,
- drift markers,
- actual refit markers when the backend records a discrete refit,
- shaded post-drift adaptation bands.

### E. Responsible AI
Use this to explain four evidence groups:
- fairness,
- explainability,
- sustainability,
- faithfulness.

Important explanation:
- fairness/XAI/sustainability on this page come from the **current run**,
- AEF comes from the **Phase 8 development evidence layer**.

### F. Copilot Workspace
Use this to demonstrate HCAI.

Show the flow:
1. user writes a natural-language goal,
2. Copilot proposes a reviewable configuration,
3. evidence keys are attached,
4. a human must approve / edit / reject it.

### G. Faithfulness Lab
Use this for a research-focused explanation of Phase 8.

Key message:
- AEF is based on **external evidence interventions**, not internal attention attribution.

### H. Export Center
Use this at the end of the demo.

Show that the UI can export:
- evidence JSON,
- metrics CSV,
- HTML report,
- reproducibility ZIP.

Important integrity statement:
- raw dataset rows are excluded,
- the frozen 23-dataset held-out split is not used by the UI.

### I. Advanced Labs
Use this only if your supervisor asks for the older specialist pages.

This page keeps the validated legacy labs accessible.

---

## 6. Suggested supervisor demo script

1. Open **Command Center** and explain the integrated architecture.
2. Open **Run Studio** and load **Synthetic drift**.
3. Run the experiment.
4. Open **3D Decision Space** and explain pre-run framework recommendation.
5. Open **Streaming Observatory** and show drift markers, post-drift adaptation bands, any recorded refit markers, and recovery evidence.
6. Open **Responsible AI** and explain fairness, XAI, sustainability, and AEF.
7. Open **Copilot Workspace** and show human review.
8. Open **Faithfulness Lab** and explain counterfactual evidence sensitivity.
9. Open **Export Center** and show reproducibility output.

---

## 7. Important scientific boundaries

Phase 9 does **not**:
- invent decorative benchmark values,
- send raw dataset rows in the export bundle,
- use the frozen 23-dataset held-out split,
- label pre-run predictions as observed outcomes,
- label AEF as a current-run fairness/XAI metric.


## 8. Dutch Census testing dataset

For a real-data functional test, use the cleaned file bundled with Phase 9:

`data/demo/dutch_census_stream_awareml.csv`

It contains **18,438 rows** and **12 columns** after removal of the leakage column.

Recommended settings:

- **Target:** `occupation_binary`
- **Sensitive attribute:** `sex`
- **Protected attribute usage:** Audit only (exclude from model)
- **Positive label:** `1`

The original uploaded file contained a categorical `occupation` column that maps exactly to `occupation_binary`; it is therefore excluded from the cleaned testing copy to prevent target leakage.

Use **Synthetic drift** when you want to demonstrate known drift behavior. Use **Dutch Census** when you want to demonstrate real tabular upload, fairness auditing, recommendation, Copilot, Responsible AI analysis, and export. The Dutch CSV has no explicit timestamp, so its existing row order should be described as the processing order rather than as a verified chronological population stream unless the original dataset documentation establishes that.


## Generic dataset setup assistant

Run Studio no longer displays a Dutch-specific setup guide. After any dataset is loaded,
AwareML analyzes the active schema and shows advisory suggestions for:

- likely target columns,
- possible fairness-audit attributes,
- time/order columns,
- possible target leakage,
- near-unique identifiers.

These suggestions never automatically alter the dataset or enable protected attributes.
The user must confirm the target and sensitive attribute using domain knowledge.
