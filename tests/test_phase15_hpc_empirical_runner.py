from pathlib import Path

from hpc.production.phase15.common import (
    EXPECTED_CASE_COUNT,
    EXPECTED_MODEL,
    EXPECTED_MODEL_DIGEST,
    EXPECTED_OLLAMA_VERSION,
    EXPECTED_PROMPT_VERSION,
    GENERATION_OPTIONS,
    controlled_cases,
    expected_task_rows,
    find_project_root,
)


ROOT = find_project_root()


def test_hpc_protocol_has_exact_24_controlled_cases():
    cases = controlled_cases()
    assert len(cases) == EXPECTED_CASE_COUNT == 24

    expected_ids = []
    for index in range(1, 7):
        for suffix in ("B", "E", "F_XAI", "F_CHAT"):
            expected_ids.append(
                "P15_{:02d}_{}".format(index, suffix)
            )
    assert [case.case_id for case in cases] == expected_ids


def test_hpc_task_manifest_rows_are_unique():
    rows = expected_task_rows(controlled_cases())
    assert [row["task_id"] for row in rows] == list(range(24))
    assert len({row["case_id"] for row in rows}) == 24
    assert all(len(row["case_sha256"]) == 64 for row in rows)


def test_runtime_and_generation_are_locked():
    assert EXPECTED_MODEL == "llama3:8b"
    assert (
        EXPECTED_MODEL_DIGEST
        == "365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1"
    )
    assert EXPECTED_OLLAMA_VERSION == "0.32.14"
    assert EXPECTED_PROMPT_VERSION == "phase15_explanation_prompt_v4"
    assert GENERATION_OPTIONS == {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "num_predict": 512,
    }


def test_slurm_array_job_is_gpu_and_24_cases():
    text = (
        ROOT
        / "hpc"
        / "production"
        / "phase15"
        / "phase15_llm_array.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --array=0-23%4" in text
    assert "#SBATCH --gres=gpu:a100-40g:1" in text
    assert "--strict-runtime" in text
    assert "SLURM_ARRAY_TASK_ID" in text


def test_resume_and_preservation_contract():
    runner = (
        ROOT
        / "hpc"
        / "production"
        / "phase15"
        / "run_phase15_case.py"
    ).read_text(encoding="utf-8")
    resume = (
        ROOT
        / "hpc"
        / "production"
        / "phase15"
        / "resume_phase15_campaign.py"
    ).read_text(encoding="utf-8")

    assert "attempts" in runner
    assert "SUCCESS.json" in runner
    assert "LAST_FAILURE.json" in runner
    assert "result_sha256" in runner
    assert "success_valid" in resume


def test_reducer_requires_complete_24_by_default():
    reducer = (
        ROOT
        / "hpc"
        / "reduce"
        / "phase15"
        / "collect_phase15_results.py"
    ).read_text(encoding="utf-8")

    assert "--allow-partial" in reducer
    assert "EXPECTED_CASE_COUNT" in reducer
    assert "phase15_empirical_summary.json" in reducer
    assert "summary_by_source.csv" in reducer
    assert "collection_manifest.json" in reducer
