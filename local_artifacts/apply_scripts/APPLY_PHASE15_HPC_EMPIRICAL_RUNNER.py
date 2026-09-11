from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


ROOT = Path.cwd()
PAYLOAD = ROOT / "phase15_hpc_payload"
BACKUP = (
    ROOT
    / ".phase15_hpc_runner_backup"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


def backup(path: Path):
    if not path.exists():
        return
    target = BACKUP / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def main():
    if not (ROOT / "awareml").exists() or not (ROOT / "hpc").exists():
        raise RuntimeError(
            "Run this installer from the AwareML_Extension project root."
        )
    if not PAYLOAD.exists():
        raise RuntimeError(
            "phase15_hpc_payload is missing. Extract the complete ZIP first."
        )

    BACKUP.mkdir(parents=True, exist_ok=True)

    for source in PAYLOAD.rglob("*"):
        if not source.is_file():
            continue
        destination = ROOT / source.relative_to(PAYLOAD)
        backup(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    gitignore = ROOT / ".gitignore"
    if gitignore.exists():
        rules = [
            "/.phase15_hpc_runner_backup/",
            "/phase15_hpc_payload/",
            "/artifacts/phase15/hpc_runs/",
            "/artifacts/phase15/hpc_setup/",
        ]
        text = gitignore.read_text(encoding="utf-8")
        missing = [rule for rule in rules if rule not in text]
        if missing:
            backup(gitignore)
            with gitignore.open("a", encoding="utf-8") as handle:
                handle.write("\n# Phase-15 HPC empirical runner artifacts\n")
                for rule in missing:
                    handle.write(rule + "\n")

    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)

    print("=" * 100)
    print("AwareML Phase-15 HPC empirical runner: APPLIED")
    print("=" * 100)
    print("Backup:", BACKUP)
    print()
    print("Installed:")
    print("  hpc/audit/phase15/")
    print("  hpc/environment_freeze/phase15/")
    print("  hpc/production/phase15/")
    print("  hpc/reduce/phase15/")
    print("  data/journal/phase15_hpc_empirical_protocol_v1/design/protocol.json")
    print("  docs/PHASE15_HPC_EMPIRICAL_RUN.md")
    print("  scripts/validate_phase15_hpc_runner.py")
    print("  tests/test_phase15_hpc_empirical_runner.py")
    print()
    print("Scientific locks:")
    print("  model: llama3:8b")
    print("  digest: 365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1")
    print("  Ollama: 0.32.14")
    print("  prompt: phase15_explanation_prompt_v4")
    print("  temperature=0, top_p=1, seed=42, num_predict=512")
    print()
    print("Run locally before pushing:")
    print("  pytest -q .\\tests\\test_phase15_hpc_empirical_runner.py")
    print("  python -m scripts.validate_phase15_hpc_runner")
    print("  pytest -q")
    print("=" * 100)


if __name__ == "__main__":
    main()
