from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from .autostreamml import AutoStreamMLAdapter
from .autoclass import AutoClassAdapter
from .evoautoml import EvoAutoMLAdapter
from .oaml import OAMLAdapter
from .chacha import ChaChaAdapter


framework_registry = OrderedDict([
    ("AutoStreamML", AutoStreamMLAdapter),
    ("AutoClass", AutoClassAdapter),
    ("EvoAutoML", EvoAutoMLAdapter),
    ("OAML", OAMLAdapter),
    ("ChaCha", ChaChaAdapter),
])


def create_frameworks(names: Iterable[str] | None = None, seed: int = 42):
    names = list(names) if names is not None else list(framework_registry.keys())
    unknown = [n for n in names if n not in framework_registry]
    if unknown:
        raise ValueError(f"Unknown framework(s): {unknown}")
    return [framework_registry[name](seed=seed) for name in names]
