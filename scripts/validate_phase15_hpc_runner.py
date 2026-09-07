from __future__ import annotations

from pathlib import Path

from hpc.production.phase15.common import (
    EXPECTED_CASE_COUNT,
    EXPECTED_MODEL,
    EXPECTED_MODEL_DIGEST,
    EXPECTED_OLLAMA_VERSION,
    EXPECTED_PROMPT_VERSION,
    GENERATION_OPTIONS,
    controlled_cases,
    find_project_root,
    protocol_path,
)


def main():
    root = find_project_root()
    cases = controlled_cases()

    array_script = (
        root
        / "hpc"
        / "production"
        / "phase15"
        / "phase15_llm_array.sbatch"
    ).read_text(encoding="utf-8")
    runner = (
        root
        / "hpc"
        / "production"
        / "phase15"
        / "run_phase15_case.py"
    ).read_text(encoding="utf-8")
    resume = (
        root
        / "hpc"
        / "production"
        / "phase15"
        / "resume_phase15_campaign.py"
    ).read_text(encoding="utf-8")
    reducer = (
        root
        / "hpc"
        / "reduce"
        / "phase15"
        / "collect_phase15_results.py"
    ).read_text(encoding="utf-8")

    checks = {
        "exact_24_controlled_cases": len(cases) == EXPECTED_CASE_COUNT,
        "model_llama3_8b": EXPECTED_MODEL == "llama3:8b",
        "digest_locked": len(EXPECTED_MODEL_DIGEST) == 64,
        "ollama_0_32_14": EXPECTED_OLLAMA_VERSION == "0.32.14",
        "prompt_v4": EXPECTED_PROMPT_VERSION == "phase15_explanation_prompt_v4",
        "deterministic_generation": GENERATION_OPTIONS == {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "num_predict": 512,
        },
        "protocol_exists": protocol_path(root).exists(),
        "slurm_array_24": "#SBATCH --array=0-23%4" in array_script,
        "gpu_partition": "#SBATCH --partition=gpu" in array_script,
        "strict_runtime": "--strict-runtime" in array_script,
        "per_case_success_marker": "SUCCESS.json" in runner,
        "append_only_attempts": "/ \"attempts\" /" in runner,
        "resume_support": "success_valid" in resume,
        "reducer_requires_24": "EXPECTED_CASE_COUNT" in reducer,
        "raw_results_not_auto_frozen": "not auto-frozen" in reducer,
    }

    print("=" * 96)
    print("Phase-15 HPC empirical runner validation")
    print("=" * 96)
    failed = []
    for name, ok in checks.items():
        print("{:<64} {}".format(name, "PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)

    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))

    print("=" * 96)
    print("Phase-15 HPC empirical runner: PASS")


if __name__ == "__main__":
    main()
