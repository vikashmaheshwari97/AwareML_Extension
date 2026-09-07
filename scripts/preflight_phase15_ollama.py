from __future__ import annotations

import subprocess
import sys
import time

from awareml.llm.client import OllamaClient


def _status(timeout=5):
    try:
        client = OllamaClient(timeout_sec=timeout)
        return client, client.status(), None
    except Exception as exc:
        return None, None, exc


def main():
    client, status, error = _status(timeout=5)

    if error is not None or not (status or {}).get("reachable"):
        print("=" * 92)
        print("Phase-15 Ollama preflight: SERVER NOT REACHABLE")
        print("=" * 92)
        print("The Phase-15 code is installed correctly, but localhost:11434")
        print("is not accepting connections.")
        if error is not None:
            print("Error:", type(error).__name__, str(error))
        elif status:
            print("Status error:", status.get("error"))
        print()
        print("Start Ollama in another PowerShell:")
        print("  ollama serve")
        print()
        print("Or start it in the background:")
        print('  Start-Process ollama -ArgumentList "serve"')
        print()
        print("Then rerun:")
        print("  python -m scripts.preflight_phase15_ollama")
        raise SystemExit(2)

    print("=" * 92)
    print("Phase-15 Ollama preflight: PASS")
    print("=" * 92)
    print("Configured model:", status.get("configured_model"))
    print("Resolved model:", status.get("resolved_model"))
    print("Available models:", status.get("models"))
    print("Server reachable:", status.get("reachable"))

    resolved = status.get("resolved_model")
    if not resolved:
        print()
        print("No usable model was resolved. Install/pull the intended model before running Phase 15.")
        raise SystemExit(3)

    print()
    print("Ready for:")
    print(
        "  python -m scripts.run_phase15_llm_benchmark "
        "--max-cases 1 --timeout-sec 300 --retries 1"
    )


if __name__ == "__main__":
    main()
