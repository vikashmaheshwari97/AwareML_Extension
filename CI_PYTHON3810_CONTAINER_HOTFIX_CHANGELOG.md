# CI Exact-Python Hotfix

## Root cause

GitHub Actions `actions/setup-python@v5` could not provision CPython 3.8.10 on
the `ubuntu-22.04` runner because that exact patch release is absent from the
current actions/python-versions manifest for Ubuntu 22.04.

The CI jobs therefore failed before dependency installation and before pytest.

## Change

Replaced `actions/setup-python@v5` with a job-level Docker Official Image:

`python:3.8.10-buster`

for both:
- `main-stack`
- `oaml-river-compatibility`

Each job explicitly asserts `sys.version_info[:3] == (3, 8, 10)`.

## Scientific/reproducibility boundary

This does not change the AwareML runtime requirement. It changes only how the
GitHub-hosted CI runner obtains the already-frozen CPython 3.8.10 interpreter.
