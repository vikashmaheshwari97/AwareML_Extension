from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from .configurable import (
    ConfigurableAutoStreamMLAdapter,
    ConfigurableAutoClassAdapter,
    ConfigurableEvoAutoMLAdapter,
    ConfigurableOAMLAdapter,
    ConfigurableChaChaAdapter,
)

framework_registry = OrderedDict([
    ("AutoStreamML", ConfigurableAutoStreamMLAdapter),
    ("AutoClass", ConfigurableAutoClassAdapter),
    ("EvoAutoML", ConfigurableEvoAutoMLAdapter),
    ("OAML", ConfigurableOAMLAdapter),
    ("ChaCha", ConfigurableChaChaAdapter),
])


def create_frameworks(
    names: Iterable[str] | None = None,
    seed: int = 42,
    configs: dict | None = None,
):
    names = list(names) if names is not None else list(framework_registry.keys())
    unknown = [n for n in names if n not in framework_registry]
    if unknown:
        raise ValueError(f"Unknown framework(s): {unknown}")
    configs = configs or {}
    return [
        framework_registry[name](seed=seed, config=dict(configs.get(name) or {}))
        for name in names
    ]
