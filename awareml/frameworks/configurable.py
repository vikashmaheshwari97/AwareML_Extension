from __future__ import annotations

"""Configurable wrappers around the validated AwareML framework adapters.

The wrappers keep current default behaviour when no advanced configuration is
supplied, while exposing the algorithm/search choices that were present in the
legacy AwareML prototype. Unsupported algorithms are never silently invented:
optional River learners are filtered at runtime and reported in get_params().
"""

import copy
import os
from typing import Any

from river import ensemble, linear_model, naive_bayes, preprocessing, tree
try:
    from river import neighbors
except Exception:
    neighbors = None

from .autostreamml import AutoStreamMLAdapter
from .autoclass import AutoClassAdapter
from .evoautoml import EvoAutoMLAdapter
from .oaml import OAMLAdapter
from .chacha import ChaChaAdapter
from .catalog import CATALOG, default_framework_config, runtime_available_algorithms
from .common import clone_model


def _scalar_params(model, declared: dict | None = None) -> dict[str, Any]:
    try:
        raw = model._get_params() if hasattr(model, "_get_params") else {}
    except Exception:
        raw = {}
    keys = list(declared or {}) if declared else list(raw)
    out = {}
    for key in keys:
        if key not in raw:
            continue
        value = raw.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
    return out


def _set_supported(model, params: dict[str, Any]):
    if not params:
        return model
    try:
        current = model._get_params() if hasattr(model, "_get_params") else {}
    except Exception:
        current = {}
    cleaned = {k: v for k, v in params.items() if k in current}
    if not cleaned:
        return model
    try:
        return model._set_params(cleaned)
    except Exception:
        # River estimators are generally cloneable/deep-copyable. Keep the
        # validated default if a version-specific parameter is unsupported.
        return model


def _algorithm_spec(framework: str, algorithm: str) -> dict:
    return (CATALOG.get(framework, {}).get("algorithms", {}).get(algorithm) or {})


def _global_config(framework: str, config: dict | None) -> dict:
    base = default_framework_config(framework)["global"]
    supplied = dict((config or {}).get("global") or {})
    return {**base, **supplied}


def _selected_algorithms(framework: str, config: dict | None) -> list[str]:
    requested = list((config or {}).get("algorithms") or [])
    if not requested:
        requested = list(default_framework_config(framework)["algorithms"])
    available = set(runtime_available_algorithms(framework))
    selected = [name for name in requested if name in available]
    return selected or [name for name in default_framework_config(framework)["algorithms"] if name in available]


def _algo_overrides(config: dict | None, algorithm: str) -> dict:
    return dict(((config or {}).get("hyperparameters") or {}).get(algorithm) or {})


def _search_params(framework: str, algorithm: str, rng, enabled: bool, override: dict) -> dict:
    if override:
        return dict(override)
    if not enabled:
        return {}
    params = {}
    for key, spec in (_algorithm_spec(framework, algorithm).get("parameters") or {}).items():
        values = list(spec.get("search") or [])
        if values:
            params[key] = rng.choice(values)
    return params


class ConfigurableAutoStreamMLAdapter(AutoStreamMLAdapter):
    def __init__(self, seed: int = 42, config: dict | None = None):
        self._advanced_config = copy.deepcopy(config or {})
        self._global = _global_config("AutoStreamML", config)
        self._algorithm_pool = _selected_algorithms("AutoStreamML", config)
        if self._global.get("execution_strategy") == "fixed_algorithm" and self._algorithm_pool:
            self._algorithm_pool = self._algorithm_pool[:1]
        self._unavailable_algorithms = [
            a for a in list((config or {}).get("algorithms") or [])
            if a not in set(runtime_available_algorithms("AutoStreamML"))
        ]
        super().__init__(
            seed=seed,
            exploration_window=int(self._global.get("exploration_window", 500)),
            budget=int(self._global.get("budget", 8)),
            ensemble_size=int(self._global.get("ensemble_size", 3)),
        )

    def _instantiate(self, name: str):
        if name == "Perceptron":
            model = linear_model.Perceptron()
        elif name == "LogisticRegression":
            model = linear_model.LogisticRegression()
        elif name == "GaussianNB":
            model = naive_bayes.GaussianNB()
        elif name == "HoeffdingTreeClassifier":
            # Preserve the current AwareML adapter default when no override is supplied.
            model = tree.HoeffdingTreeClassifier(grace_period=100)
        elif name == "HoeffdingAdaptiveTreeClassifier" and hasattr(tree, "HoeffdingAdaptiveTreeClassifier"):
            model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=100)
        elif name == "AdaptiveRandomForestClassifier" and hasattr(ensemble, "AdaptiveRandomForestClassifier"):
            model = ensemble.AdaptiveRandomForestClassifier(n_models=5, seed=self.seed)
        elif name == "KNNClassifier" and neighbors is not None and hasattr(neighbors, "KNNClassifier"):
            model = neighbors.KNNClassifier(n_neighbors=5)
        else:
            raise ValueError("Unsupported AutoStreamML algorithm: {}".format(name))
        params = _search_params(
            "AutoStreamML",
            name,
            self._rng,
            bool(self._global.get("search_hyperparameters", False)),
            _algo_overrides(self._advanced_config, name),
        )
        return _set_supported(model, params)

    def _base_models(self):
        models = []
        for name in self._algorithm_pool:
            try:
                models.append(self._instantiate(name))
            except Exception:
                continue
        if models:
            return models
        return super()._base_models()

    def _active_model(self):
        try:
            return list(self.candidates[self.best_idx].steps.values())[-1]
        except Exception:
            return None

    def get_params(self) -> dict:
        data = super().get_params()
        model = self._active_model()
        name = type(model).__name__ if model is not None else None
        declared = (_algorithm_spec("AutoStreamML", name).get("parameters") or {}) if name else {}
        tunable = {
            "exploration_window": self.exploration_window,
            "budget": self.budget,
            "ensemble_size": self.ensemble_size,
        }
        tunable.update(_scalar_params(model, declared))
        data.update({
            "execution_strategy": self._global.get("execution_strategy", "adaptive_pool"),
            "search_hyperparameters": bool(self._global.get("search_hyperparameters", False)),
            "search_pool": list(self._algorithm_pool),
            "active_algorithm": name,
            "active_hyperparameters": _scalar_params(model, declared),
            "tunable": tunable,
            "unavailable_requested_algorithms": list(self._unavailable_algorithms),
            "configuration_source": "AwareML v6 framework control",
        })
        return data


class ConfigurableAutoClassAdapter(AutoClassAdapter):
    def __init__(self, seed: int = 42, config: dict | None = None):
        self._advanced_config = copy.deepcopy(config or {})
        self._global = _global_config("AutoClass", config)
        self._algorithm_pool = _selected_algorithms("AutoClass", config)
        if self._global.get("execution_strategy") == "fixed_algorithm" and self._algorithm_pool:
            self._algorithm_pool = self._algorithm_pool[:1]
        self._unavailable_algorithms = [
            a for a in list((config or {}).get("algorithms") or [])
            if a not in set(runtime_available_algorithms("AutoClass"))
        ]
        super().__init__(
            seed=seed,
            exploration_window=int(self._global.get("exploration_window", 500)),
            population_size=int(self._global.get("population_size", 8)),
        )
        # Restore the richer legacy AutoClass search spaces while allowing a
        # researcher to lock an exact value through Run Studio.
        spaces = {}
        search_enabled = bool(self._global.get("search_hyperparameters", True))
        for algorithm in self._algorithm_pool:
            declared = _algorithm_spec("AutoClass", algorithm).get("parameters") or {}
            override = _algo_overrides(self._advanced_config, algorithm)
            spaces[algorithm] = {}
            for key, spec in declared.items():
                if key in override:
                    spaces[algorithm][key] = [override[key]]
                elif search_enabled and spec.get("search"):
                    spaces[algorithm][key] = list(spec["search"])
                else:
                    spaces[algorithm][key] = [spec.get("default")]
        self._spaces = spaces
        self._build()

    def _model_pool(self):
        out = []
        for name in self._algorithm_pool:
            try:
                if name == "HoeffdingTreeClassifier":
                    out.append(tree.HoeffdingTreeClassifier())
                elif name == "HoeffdingAdaptiveTreeClassifier" and hasattr(tree, "HoeffdingAdaptiveTreeClassifier"):
                    out.append(tree.HoeffdingAdaptiveTreeClassifier())
                elif name == "KNNClassifier" and neighbors is not None and hasattr(neighbors, "KNNClassifier"):
                    out.append(neighbors.KNNClassifier())
                elif name == "KNNADWINClassifier" and neighbors is not None and hasattr(neighbors, "KNNADWINClassifier"):
                    out.append(neighbors.KNNADWINClassifier())
                elif name == "LogisticRegression":
                    out.append(linear_model.LogisticRegression())
                elif name == "GaussianNB":
                    out.append(naive_bayes.GaussianNB())
            except Exception:
                continue
        return out or super()._model_pool()

    def get_params(self) -> dict:
        data = super().get_params()
        model = None
        try:
            model = self.population[self.best_idx]
        except Exception:
            pass
        name = type(model).__name__ if model is not None else None
        declared = (_algorithm_spec("AutoClass", name).get("parameters") or {}) if name else {}
        active = _scalar_params(model, declared)
        tunable = {
            "exploration_window": self.exploration_window,
            "population_size": self.population_size,
        }
        tunable.update(active)
        data.update({
            "execution_strategy": self._global.get("execution_strategy", "adaptive_pool"),
            "search_hyperparameters": bool(self._global.get("search_hyperparameters", True)),
            "search_pool": list(self._algorithm_pool),
            "active_algorithm": name,
            "active_hyperparameters": active,
            "tunable": tunable,
            "unavailable_requested_algorithms": list(self._unavailable_algorithms),
            "configuration_source": "AwareML v6 framework control",
        })
        return data


class ConfigurableEvoAutoMLAdapter(EvoAutoMLAdapter):
    def __init__(self, seed: int = 42, config: dict | None = None):
        self._advanced_config = copy.deepcopy(config or {})
        self._global = _global_config("EvoAutoML", config)
        super().__init__(
            seed=seed,
            population_size=int(self._global.get("population_size", 10)),
            sampling_rate=int(self._global.get("sampling_rate", 500)),
        )

    def get_params(self) -> dict:
        data = super().get_params()
        data.update({
            "active_algorithm": "EvolutionaryBaggingClassifier",
            "search_pool": ["EvolutionaryBaggingClassifier"],
            "tunable": {
                "population_size": self.population_size,
                "sampling_rate": self.sampling_rate,
            },
            "configuration_source": "AwareML v6 framework control",
        })
        return data


class ConfigurableOAMLAdapter(OAMLAdapter):
    def __init__(self, seed: int = 42, config: dict | None = None):
        self._advanced_config = copy.deepcopy(config or {})
        self._global = _global_config("OAML", config)
        selected = list((config or {}).get("algorithms") or [])
        mode = str(self._global.get("mode") or os.getenv("AWAREML_OAML_MODE", "online")).lower()
        if "GAMA AutoPipeline" in selected:
            mode = "gama"
        online_algorithm = next(
            (a for a in selected if a in {"HoeffdingAdaptiveTreeClassifier", "HoeffdingTreeClassifier"}),
            "HoeffdingAdaptiveTreeClassifier",
        )
        values = {
            "AWAREML_OAML_MODE": mode,
            "AWAREML_OAML_ONLINE_ALGORITHM": online_algorithm,
            "AWAREML_OAML_GAMA_BUDGET": str(int(self._global.get("gama_time_budget_sec", 5))),
            "AWAREML_OAML_GAMA_WARMUP": str(int(self._global.get("gama_warmup", 50))),
            "AWAREML_OAML_GAMA_REFIT": str(int(self._global.get("gama_refit_interval", 500))),
        }
        old = {k: os.environ.get(k) for k in values}
        try:
            os.environ.update(values)
            super().__init__(seed=seed, grace_period=int(self._global.get("grace_period", 100)))
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self._selected_online_algorithm = online_algorithm

    def get_params(self) -> dict:
        data = super().get_params()
        active = "GAMA AutoPipeline" if str(data.get("mode")) == "gama" else data.get("online_model", self._selected_online_algorithm)
        data.update({
            "active_algorithm": active,
            "search_pool": [active],
            "tunable": {
                "grace_period": int(self._global.get("grace_period", 100)),
                "gama_time_budget_sec": int(self._global.get("gama_time_budget_sec", 5)),
                "gama_warmup": int(self._global.get("gama_warmup", 50)),
                "gama_refit_interval": int(self._global.get("gama_refit_interval", 500)),
            },
            "configuration_source": "AwareML v6 framework control",
        })
        return data


class ConfigurableChaChaAdapter(ChaChaAdapter):
    def __init__(self, seed: int = 42, config: dict | None = None):
        self._advanced_config = copy.deepcopy(config or {})
        self._global = _global_config("ChaCha", config)
        super().__init__(seed=seed, exploration=float(self._global.get("exploration", 1.2)))

    def _new_autovw(self, seed: int):
        if self._AutoVW is None or self._loguniform is None:
            raise RuntimeError("FLAML AutoVW backend is unavailable.")
        l1 = max(1e-12, float(self._global.get("l1", 1e-8)))
        l2 = max(1e-12, float(self._global.get("l2", 1e-6)))
        lr = float(self._global.get("learning_rate_init", 0.1))
        search_space = {
            "learning_rate": self._loguniform(lower=max(0.001, lr / 20.0), upper=min(1.0, max(0.01, lr * 8.0))),
            "l1": self._loguniform(lower=max(1e-12, l1 / 100.0), upper=max(1e-10, l1 * 100.0)),
            "l2": self._loguniform(lower=max(1e-12, l2 / 100.0), upper=max(1e-10, l2 * 100.0)),
            "loss_function": "logistic",
            "link": "logistic",
        }
        return self._AutoVW(
            max_live_model_num=int(self._global.get("max_live_model_num", 4)),
            search_space=search_space,
            init_config={
                "learning_rate": lr,
                "l1": l1,
                "l2": l2,
                "loss_function": "logistic",
                "link": "logistic",
            },
            metric="mae_clipped",
            random_seed=int(seed),
            model_selection_mode="min",
        )

    def get_params(self) -> dict:
        data = super().get_params()
        data.update({
            "active_algorithm": "AutoVW" if data.get("native_autovw_active") else "transparent-UCB-fallback",
            "search_pool": ["AutoVW"],
            "tunable": {
                "max_live_model_num": int(self._global.get("max_live_model_num", 4)),
                "learning_rate_init": float(self._global.get("learning_rate_init", 0.1)),
                "l1": float(self._global.get("l1", 1e-8)),
                "l2": float(self._global.get("l2", 1e-6)),
                "exploration": float(self.exploration),
            },
            "configuration_source": "AwareML v6 framework control",
        })
        return data
