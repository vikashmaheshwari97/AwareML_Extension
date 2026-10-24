"""Resolve relocated text resources without changing their frozen content.

Historical protocols retain their original .txt paths. The repository stores
runtime prompts as .prompt and active pointers as .ref to keep them versioned
under its documentation policy. Explicit existing paths always take priority.
"""
from pathlib import Path


def resolve_text_resource(path):
    path = Path(path)
    if path.is_file():
        return path
    if path.suffix == ".txt":
        for suffix in (".prompt", ".ref"):
            candidate = path.with_suffix(suffix)
            if candidate.is_file():
                return candidate
    return path
