from __future__ import annotations

"""Research-facing framework/algorithm configuration catalogue.

The catalogue exposes only controls that are actually supported by the current
AwareML adapters. It deliberately distinguishes framework-level search controls
from algorithm-level hyperparameters and keeps unsupported legacy options out of
an executable configuration.
"""

from copy import deepcopy


CATALOG = {
    "AutoStreamML": {
        "description": "Adaptive streaming pipeline search with a configurable learner pool.",
        "strategy": "pool",
        "default_algorithms": [
            "LogisticRegression", "GaussianNB", "HoeffdingTreeClassifier", "KNNClassifier"
        ],
        "global": {
            "exploration_window": {"type": "int", "default": 500, "min": 100, "max": 5000, "step": 100},
            "budget": {"type": "int", "default": 8, "min": 3, "max": 30, "step": 1},
            "ensemble_size": {"type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
            "execution_strategy": {"type": "choice", "default": "adaptive_pool", "choices": ["adaptive_pool", "fixed_algorithm"]},
            "search_hyperparameters": {"type": "bool", "default": False},
        },
        "algorithms": {
            "Perceptron": {
                "parameters": {"l2": {"type": "float", "default": 0.001, "min": 0.0, "max": 0.05, "step": 0.001, "search": [0.0, 0.001, 0.005, 0.01]}}
            },
            "LogisticRegression": {
                "parameters": {"l2": {"type": "float", "default": 0.001, "min": 0.0, "max": 0.05, "step": 0.001, "search": [0.0, 0.0001, 0.001, 0.01]}}
            },
            "GaussianNB": {"parameters": {}},
            "HoeffdingTreeClassifier": {
                "parameters": {
                    "max_depth": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5, "search": [20, 50, 100]},
                    "grace_period": {"type": "int", "default": 100, "min": 20, "max": 1000, "step": 20, "search": [50, 100, 200, 500]},
                    "delta": {"type": "float", "default": 1e-7, "min": 1e-9, "max": 0.05, "step": 1e-5, "search": [1e-7, 1e-5, 1e-3, 1e-2]},
                    "tau": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01, "search": [0.02, 0.05, 0.1]},
                    "nb_threshold": {"type": "int", "default": 10, "min": 0, "max": 100, "step": 5, "search": [0, 10, 30, 50]},
                    "split_criterion": {"type": "choice", "default": "info_gain", "choices": ["info_gain", "gini", "hellinger"], "search": ["info_gain", "gini", "hellinger"]},
                    "leaf_prediction": {"type": "choice", "default": "nba", "choices": ["mc", "nb", "nba"], "search": ["mc", "nb", "nba"]},
                }
            },
            "HoeffdingAdaptiveTreeClassifier": {
                "parameters": {
                    "max_depth": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5, "search": [20, 50, 100]},
                    "grace_period": {"type": "int", "default": 100, "min": 20, "max": 1000, "step": 20, "search": [50, 100, 200, 500]},
                    "delta": {"type": "float", "default": 1e-7, "min": 1e-9, "max": 0.05, "step": 1e-5, "search": [1e-7, 1e-5, 1e-3, 1e-2]},
                    "tau": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01, "search": [0.02, 0.05, 0.1]},
                    "nb_threshold": {"type": "int", "default": 10, "min": 0, "max": 100, "step": 5, "search": [0, 10, 30, 50]},
                    "split_criterion": {"type": "choice", "default": "info_gain", "choices": ["info_gain", "gini", "hellinger"], "search": ["info_gain", "gini", "hellinger"]},
                    "leaf_prediction": {"type": "choice", "default": "nba", "choices": ["mc", "nb", "nba"], "search": ["mc", "nb", "nba"]},
                }
            },
            "AdaptiveRandomForestClassifier": {
                "parameters": {
                    "n_models": {"type": "int", "default": 5, "min": 2, "max": 20, "step": 1, "search": [3, 5, 8, 10]},
                    "max_depth": {"type": "int", "default": 50, "min": 5, "max": 200, "step": 5, "search": [20, 50, 100]},
                    "grace_period": {"type": "int", "default": 200, "min": 20, "max": 1000, "step": 20, "search": [50, 100, 200, 500]},
                    "lambda_value": {"type": "float", "default": 6.0, "min": 1.0, "max": 20.0, "step": 1.0, "search": [2.0, 4.0, 6.0, 10.0]},
                    "delta": {"type": "float", "default": 1e-7, "min": 1e-9, "max": 0.05, "step": 1e-5, "search": [1e-7, 1e-5, 1e-3]},
                    "tau": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01, "search": [0.02, 0.05, 0.1]},
                    "leaf_prediction": {"type": "choice", "default": "nba", "choices": ["mc", "nb", "nba"], "search": ["mc", "nb", "nba"]},
                }
            },
            "KNNClassifier": {
                "parameters": {
                    "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 30, "step": 1, "search": [3, 5, 7, 9]},
                    "window_size": {"type": "int", "default": 1000, "min": 100, "max": 10000, "step": 100, "search": [500, 1000, 2500, 5000]},
                    "weighted": {"type": "bool", "default": False, "search": [False, True]},
                    "p": {"type": "int", "default": 2, "min": 1, "max": 5, "step": 1, "search": [1, 2, 3]},
                }
            },
        },
    },
    "AutoClass": {
        "description": "Adaptive population search with legacy AutoClass learner choices restored as optional controls.",
        "strategy": "pool",
        "default_algorithms": ["HoeffdingTreeClassifier", "LogisticRegression", "GaussianNB"],
        "global": {
            "exploration_window": {"type": "int", "default": 500, "min": 100, "max": 5000, "step": 100},
            "population_size": {"type": "int", "default": 8, "min": 3, "max": 30, "step": 1},
            "execution_strategy": {"type": "choice", "default": "adaptive_pool", "choices": ["adaptive_pool", "fixed_algorithm"]},
            "search_hyperparameters": {"type": "bool", "default": True},
        },
        "algorithms": {
            "HoeffdingTreeClassifier": {
                "parameters": {
                    "grace_period": {"type": "int", "default": 100, "min": 20, "max": 1000, "step": 20, "search": [50, 100, 200, 300]},
                    "delta": {"type": "float", "default": 1e-7, "min": 1e-9, "max": 0.05, "step": 1e-5, "search": [1e-7, 1e-5, 1e-3, 1e-2]},
                    "tau": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01, "search": [0.02, 0.05, 0.1]},
                    "leaf_prediction": {"type": "choice", "default": "nba", "choices": ["mc", "nb", "nba"], "search": ["mc", "nb", "nba"]},
                }
            },
            "HoeffdingAdaptiveTreeClassifier": {
                "parameters": {
                    "grace_period": {"type": "int", "default": 100, "min": 20, "max": 1000, "step": 20, "search": [50, 100, 200, 300]},
                    "delta": {"type": "float", "default": 1e-7, "min": 1e-9, "max": 0.05, "step": 1e-5, "search": [1e-7, 1e-5, 1e-3, 1e-2]},
                    "tau": {"type": "float", "default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01, "search": [0.02, 0.05, 0.1]},
                    "leaf_prediction": {"type": "choice", "default": "nba", "choices": ["mc", "nb", "nba"], "search": ["mc", "nb", "nba"]},
                }
            },
            "KNNClassifier": {
                "parameters": {"n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 30, "step": 1, "search": [3, 5, 7, 9]}}
            },
            "KNNADWINClassifier": {
                "parameters": {"n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 30, "step": 1, "search": [3, 5, 7, 9]}}
            },
            "LogisticRegression": {
                "parameters": {"l2": {"type": "float", "default": 0.001, "min": 0.0, "max": 0.05, "step": 0.001, "search": [0.0, 0.0001, 0.001, 0.01]}}
            },
            "GaussianNB": {"parameters": {}},
        },
    },
    "EvoAutoML": {
        "description": "Native EvOAutoML evolutionary ensemble.",
        "strategy": "fixed",
        "default_algorithms": ["EvolutionaryBaggingClassifier"],
        "global": {
            "population_size": {"type": "int", "default": 10, "min": 4, "max": 50, "step": 1},
            "sampling_rate": {"type": "int", "default": 500, "min": 25, "max": 5000, "step": 25},
        },
        "algorithms": {"EvolutionaryBaggingClassifier": {"parameters": {}}},
    },
    "OAML": {
        "description": "Online River learner or GAMA AutoPipeline.",
        "strategy": "mode",
        "default_algorithms": ["HoeffdingAdaptiveTreeClassifier"],
        "global": {
            "mode": {"type": "choice", "default": "online", "choices": ["online", "gama"]},
            "grace_period": {"type": "int", "default": 100, "min": 20, "max": 1000, "step": 20},
            "gama_time_budget_sec": {"type": "int", "default": 5, "min": 2, "max": 120, "step": 1},
            "gama_warmup": {"type": "int", "default": 50, "min": 20, "max": 5000, "step": 10},
            "gama_refit_interval": {"type": "int", "default": 500, "min": 50, "max": 10000, "step": 50},
        },
        "algorithms": {
            "HoeffdingAdaptiveTreeClassifier": {"parameters": {}},
            "HoeffdingTreeClassifier": {"parameters": {}},
            "GAMA AutoPipeline": {"parameters": {}},
        },
    },
    "ChaCha": {
        "description": "Native FLAML AutoVW/ChaCha scheduler.",
        "strategy": "fixed",
        "default_algorithms": ["AutoVW"],
        "global": {
            "max_live_model_num": {"type": "int", "default": 4, "min": 2, "max": 20, "step": 1},
            "learning_rate_init": {"type": "float", "default": 0.1, "min": 0.005, "max": 0.8, "step": 0.005},
            "l1": {"type": "float", "default": 1e-8, "min": 0.0, "max": 0.01, "step": 1e-7},
            "l2": {"type": "float", "default": 1e-6, "min": 0.0, "max": 0.05, "step": 1e-6},
            "exploration": {"type": "float", "default": 1.2, "min": 0.0, "max": 5.0, "step": 0.1},
        },
        "algorithms": {"AutoVW": {"parameters": {}}},
    },
}


def framework_catalog():
    return deepcopy(CATALOG)


def default_framework_config(name: str) -> dict:
    spec = CATALOG[name]
    return {
        "algorithms": list(spec.get("default_algorithms") or []),
        "global": {k: v.get("default") for k, v in (spec.get("global") or {}).items()},
        "hyperparameters": {},
    }


def _has(name, container):
    return bool(container is not None and hasattr(container, name))


def runtime_available_algorithms(framework: str) -> list[str]:
    """Filter optional main-environment River algorithms without importing workers."""
    names = list(CATALOG[framework]["algorithms"].keys())
    if framework not in {"AutoStreamML", "AutoClass"}:
        return names
    try:
        from river import tree, linear_model, naive_bayes, ensemble
        try:
            from river import neighbors
        except Exception:
            neighbors = None
    except Exception:
        return [n for n in names if n in {"LogisticRegression", "GaussianNB", "HoeffdingTreeClassifier"}]
    lookup = {
        "Perceptron": _has("Perceptron", linear_model),
        "LogisticRegression": _has("LogisticRegression", linear_model),
        "GaussianNB": _has("GaussianNB", naive_bayes),
        "HoeffdingTreeClassifier": _has("HoeffdingTreeClassifier", tree),
        "HoeffdingAdaptiveTreeClassifier": _has("HoeffdingAdaptiveTreeClassifier", tree),
        "AdaptiveRandomForestClassifier": _has("AdaptiveRandomForestClassifier", ensemble),
        "KNNClassifier": _has("KNNClassifier", neighbors),
        "KNNADWINClassifier": _has("KNNADWINClassifier", neighbors),
    }
    return [n for n in names if lookup.get(n, True)]
