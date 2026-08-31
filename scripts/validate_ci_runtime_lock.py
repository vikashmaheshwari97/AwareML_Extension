from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
EXPECTED_IMAGE = "python:3.8.10-buster"
EXPECTED_JOBS = ("main-stack", "oaml-river-compatibility")


def main() -> None:
    if not WORKFLOW.exists():
        raise RuntimeError("Missing CI workflow: {}".format(WORKFLOW))

    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") or {}

    for job_name in EXPECTED_JOBS:
        if job_name not in jobs:
            raise RuntimeError("Missing CI job: {}".format(job_name))

        job = jobs[job_name]
        image = (job.get("container") or {}).get("image")
        if image != EXPECTED_IMAGE:
            raise RuntimeError(
                "{} must use exact container {}; found {}".format(
                    job_name, EXPECTED_IMAGE, image
                )
            )

        serialized = str(job)
        if "actions/setup-python" in serialized:
            raise RuntimeError(
                "{} still depends on actions/setup-python; exact 3.8.10 "
                "must come from the container image.".format(job_name)
            )

        steps = job.get("steps") or []
        verify_steps = [
            step for step in steps
            if "Verify exact CPython 3.8.10" == step.get("name")
        ]
        if len(verify_steps) != 1:
            raise RuntimeError(
                "{} must contain exactly one exact-interpreter verification step.".format(
                    job_name
                )
            )

        command = str(verify_steps[0].get("run") or "")
        if "(3, 8, 10)" not in command:
            raise RuntimeError(
                "{} interpreter verification does not assert 3.8.10.".format(
                    job_name
                )
            )

    print("=" * 72)
    print("AwareML CI exact-Python validation: PASS")
    print("=" * 72)
    print("Runtime source: Docker Official Image")
    print("Image:", EXPECTED_IMAGE)
    print("Jobs:", list(EXPECTED_JOBS))
    print("actions/setup-python used: False")
    print("Required interpreter: CPython 3.8.10")
    print("=" * 72)


if __name__ == "__main__":
    main()
