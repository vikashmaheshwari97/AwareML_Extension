import json

from awareml.experiments import ExperimentStore, RunSummaryRecord, WindowMetricRecord


def test_per_experiment_storage_is_append_safe(tmp_path):
    store = ExperimentStore(str(tmp_path / "meta"))
    run = RunSummaryRecord(
        experiment_id="adult__autostreamml__s42", protocol_version="meta-v2",
        dataset_id="adult", framework="AutoStreamML", seed=42, status="ok",
        sustainability_status="not_measured",
    )
    run_path = store.write_run_summary(run)
    assert run_path.exists()

    for i in range(2):
        store.append_window(WindowMetricRecord(
            experiment_id=run.experiment_id,
            window_id=i, sample_index=(i + 1) * 500,
            prequential_accuracy=0.8 + i * 0.01,
            prequential_macro_f1=0.79 + i * 0.01,
        ))
    lines = (run_path.parent / "windows.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["sample_index"] == 1000
