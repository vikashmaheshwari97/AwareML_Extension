from __future__ import annotations

import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("=" * 72)
    print("AwareML local main-stack CI check")
    print("=" * 72)

    # Do not compile the local .venv. GitHub Actions does not keep its environment
    # inside the repository, so compiling .venv locally is not CI-equivalent.
    targets = [ROOT / "awareml", ROOT / "scripts", ROOT / "tests"]
    ok = True
    for target in targets:
        if target.exists():
            print("Compiling:", target.relative_to(ROOT))
            ok = compileall.compile_dir(str(target), quiet=1, force=False) and ok

    app = ROOT / "app.py"
    if app.exists():
        ok = compileall.compile_file(str(app), quiet=1, force=False) and ok

    if not ok:
        raise SystemExit("Repository source compilation failed.")

    print("Compilation: PASS")
    print("")
    print("Running full pytest suite...")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT))
    if result.returncode:
        raise SystemExit(result.returncode)

    print("")
    print("=" * 72)
    print("LOCAL MAIN-STACK: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
