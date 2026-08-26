# Dutch Census testing guide for AwareML

## Dataset supplied for testing

The uploaded CSV contains:

- **18,438 rows**
- **13 original columns**
- **0 missing values**
- binary target candidate: **`occupation_binary`**
- sensitive attribute candidate: **`sex`**

Target distribution:

- `occupation_binary = 0`: **8,309** rows
- `occupation_binary = 1`: **10,129** rows

Sensitive-attribute distribution:

- `sex = 1`: **9,174** rows
- `sex = 2`: **9,264** rows

## Important leakage issue

Do **not** use the original `occupation` column as a model feature when predicting `occupation_binary`.

In the supplied file the mapping is exact:

- `occupation = 2_1` → `occupation_binary = 0`
- `occupation = 5_4_9` → `occupation_binary = 1`

Keeping `occupation` would therefore expose the target directly and make the test scientifically invalid.

Phase 9 includes a cleaned copy:

`data/demo/dutch_census_stream_awareml.csv`

This copy removes only `occupation` and keeps `occupation_binary` as the target.

## Recommended AwareML settings

In **Run Studio**:

1. Choose **Upload CSV**.
2. Upload `data/demo/dutch_census_stream_awareml.csv`.
3. Select target: **`occupation_binary`**.
4. Select sensitive attribute: **`sex`**.
5. Choose protected attribute usage: **Audit only (exclude from model)** for the first fairness test.
6. Select positive label: **`1`**.
7. Use the normal shared temporal protocol.
8. Run the desired frameworks.

## How to describe the dataset in the demo

This dataset is useful as a **functional real-data testing stream** for the AwareML pipeline.

The CSV does not contain an explicit timestamp. AwareML will process the existing row order as stream order. Therefore, unless the source documentation independently establishes that this row ordering is chronological, do not claim that detected changes are verified real temporal concept drift in the Dutch population. For a true drift demonstration, the synthetic drift stream remains the cleaner example.

## Recommended supervisor demonstration

Use two datasets for two different purposes:

### Synthetic drift
Use it to show:

- known streaming behavior,
- drift markers,
- post-drift adaptation visualization,
- temporal accuracy/F1/latency.

### Dutch Census
Use it to show:

- uploaded real tabular data,
- binary classification,
- sensitive-attribute auditing,
- framework recommendation,
- Responsible AI comparison,
- Copilot configuration,
- reproducibility export.
