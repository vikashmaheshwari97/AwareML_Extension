from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .schemas import ConfigDiffItem


def _flatten(
    value: Any,
    prefix: str = "",
) -> Dict[str, Any]:
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = (
                "{}.{}".format(
                    prefix,
                    key,
                )
                if prefix
                else str(key)
            )
            out.update(
                _flatten(
                    child,
                    path,
                )
            )
        return out
    out[prefix] = value
    return out


def diff_configs(
    before: Any,
    after: Any,
) -> List[ConfigDiffItem]:
    if hasattr(before, "model_dump"):
        before = before.model_dump()
    if hasattr(after, "model_dump"):
        after = after.model_dump()

    before = before or {}
    after = after or {}

    left = _flatten(before)
    right = _flatten(after)

    paths = sorted(
        set(left) | set(right)
    )
    return [
        ConfigDiffItem(
            path=path,
            before=left.get(path),
            after=right.get(path),
        )
        for path in paths
        if left.get(path)
        != right.get(path)
    ]


def deep_merge(
    base: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (
        patch or {}
    ).items():
        if (
            isinstance(value, dict)
            and isinstance(
                out.get(key),
                dict,
            )
        ):
            out[key] = deep_merge(
                out[key],
                value,
            )
        else:
            out[key] = deepcopy(
                value
            )
    return out
