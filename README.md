# AwareML — Historical Meta-Recommender pre-Phase-14 hotfix

Adds a second Copilot Workspace tab that can recommend a framework **without a dataset** using the frozen 705-run Meta-Dataset V2 evidence.

## New Copilot Workspace tabs

1. **Goal Copilot** — existing dataset-aware research pipeline. Natural language → LLaMA/V3.1 objectives → equal weights → dataset meta-profile → ML Recommender V2.
2. **Historical Meta-Recommender · No dataset** — new preference-only global historical prior from 47 datasets × 5 frameworks × 3 seeds.

The Historical tab is not called ML Recommender V2 because it does not have dataset meta-features. It is an empirical historical recommender over the frozen meta-evidence.

## Seed handling

- **Stable 3-seed aggregate (recommended):** uses the 235 dataset/framework profiles derived from all three seeds.
- **Best observed single seed · exploratory:** chooses one actual seed per dataset/framework using the complete weighted Accuracy/Runtime/Energy/CO2 utility. It never mixes metrics from different seeds.

## Apply

From the AwareML_Extension project root:

```powershell
Expand-Archive `
  -Path "$env:USERPROFILE\Downloads\AwareML_Historical_Meta_Recommender_Pre_Phase14_Hotfix.zip" `
  -DestinationPath "C:\Users\maheshwari\PycharmProjects\AwareML_Extension" `
  -Force

python .\APPLY_HISTORICAL_META_RECOMMENDER_PRE14_HOTFIX.py
```

Then:

```powershell
pytest -q .\tests\test_historical_meta_recommender_pre14.py
python -m scripts.validate_historical_meta_recommender_pre14
python -m scripts.validate_copilot_clarity_pre14
python -m scripts.validate_pre14_usability_upgrade
python -m scripts.validate_phase12_v31_pre14
streamlit run app.py
```

Do not git-add `.pre14_historical_meta_backup\`.
