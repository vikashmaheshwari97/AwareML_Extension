from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

HOTFIX_ID = "phase18_runtime_hotfix_v1"


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    try:
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            return value.item()
    except Exception:
        pass
    return value


def json_safe(value: Any) -> Any:
    value = _python_scalar(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(json_safe(k)): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except Exception:
        pass
    return str(value)


def install_experiment_store_json_hotfix() -> None:
    from awareml.experiments.storage import ExperimentStore
    if getattr(ExperimentStore, "_phase18_json_hotfix_installed", False):
        return
    original = ExperimentStore._jsonable
    def patched_jsonable(record: Any):
        return json_safe(original(record))
    ExperimentStore._jsonable = staticmethod(patched_jsonable)
    ExperimentStore._phase18_json_hotfix_installed = True


def _label_token(value: Any) -> str:
    value = _python_scalar(value)
    return json.dumps({"python_type": type(value).__name__, "value": value}, sort_keys=True, ensure_ascii=False, default=str)


def _ensure_evo_codec(adapter: Any) -> None:
    if not hasattr(adapter, "_phase18_label_to_code"):
        adapter._phase18_label_to_code = {}
    if not hasattr(adapter, "_phase18_code_to_label"):
        adapter._phase18_code_to_label = {}


def _encode_evo_label(adapter: Any, value: Any) -> int:
    _ensure_evo_codec(adapter)
    value = _python_scalar(value)
    token = _label_token(value)
    if token not in adapter._phase18_label_to_code:
        code = len(adapter._phase18_label_to_code)
        adapter._phase18_label_to_code[token] = code
        adapter._phase18_code_to_label[code] = value
    return int(adapter._phase18_label_to_code[token])


def _decode_evo_label(adapter: Any, value: Any) -> Any:
    _ensure_evo_codec(adapter)
    value = _python_scalar(value)
    if value is None:
        return None
    try:
        code = int(value)
    except Exception:
        return value
    return adapter._phase18_code_to_label.get(code, value)


def install_evo_label_codec_hotfix() -> None:
    from awareml.frameworks.evoautoml import EvoAutoMLAdapter
    if getattr(EvoAutoMLAdapter, "_phase18_label_codec_hotfix_installed", False):
        return
    original_init = EvoAutoMLAdapter.__init__
    original_get_params = EvoAutoMLAdapter.get_params
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._phase18_label_to_code = {}
        self._phase18_code_to_label = {}
    def patched_learn_one(self, x, y):
        self._request({"cmd": "learn", "x": x, "y": _encode_evo_label(self, y)})
    def patched_predict_one(self, x):
        raw = self._request({"cmd": "predict", "x": x}).get("prediction")
        return _decode_evo_label(self, raw)
    def decode_proba_items(self, items):
        if not isinstance(items, list):
            return None
        out = {}
        for pair in items:
            if isinstance(pair, list) and len(pair) == 2:
                try:
                    out[_decode_evo_label(self, pair[0])] = float(pair[1])
                except Exception:
                    continue
        return out or None
    def patched_predict_proba_one(self, x):
        raw = self._request({"cmd": "predict_proba", "x": x}).get("proba")
        return decode_proba_items(self, raw)
    def patched_predict(self, X):
        rows = X.to_dict(orient="records") if isinstance(X, pd.DataFrame) else list(X)
        raw = self._request({"cmd": "predict_batch", "rows": rows}).get("predictions", [])
        return np.asarray([_decode_evo_label(self, v) for v in raw], dtype=object)
    def patched_predict_proba(self, X):
        rows = X.to_dict(orient="records") if isinstance(X, pd.DataFrame) else list(X)
        raw = self._request({"cmd": "predict_proba_batch", "rows": rows}).get("probas", [])
        out = [decode_proba_items(self, items) for items in raw]
        if any(not isinstance(v, dict) or not v for v in out):
            raise RuntimeError("EvoAutoML class probabilities are unavailable for one or more rows.")
        return out
    def patched_get_params(self):
        params = dict(original_get_params(self))
        params["phase18_external_label_codec"] = HOTFIX_ID
        params["phase18_label_codec_classes_seen"] = len(getattr(self, "_phase18_label_to_code", {}))
        return params
    EvoAutoMLAdapter.__init__ = patched_init
    EvoAutoMLAdapter.learn_one = patched_learn_one
    EvoAutoMLAdapter.predict_one = patched_predict_one
    EvoAutoMLAdapter.predict_proba_one = patched_predict_proba_one
    EvoAutoMLAdapter.predict = patched_predict
    EvoAutoMLAdapter.predict_proba = patched_predict_proba
    EvoAutoMLAdapter.get_params = patched_get_params
    EvoAutoMLAdapter._phase18_label_codec_hotfix_installed = True


def install_phase18_runtime_hotfixes() -> None:
    install_experiment_store_json_hotfix()
    install_evo_label_codec_hotfix()
