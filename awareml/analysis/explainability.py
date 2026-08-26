from __future__ import annotations

from typing import Any, Iterable, Optional
import numpy as np
import pandas as pd

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None


def _safe_predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict with a batch-style or River-style streaming estimator."""
    try:
        values = model.predict(X)
        return np.asarray(values, dtype=object).reshape(-1)
    except Exception:
        rows = X.to_dict(orient="records")
        return np.asarray([model.predict_one(r) for r in rows], dtype=object)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_accuracy(y_true, y_pred) -> float:
    """Accuracy for heterogeneous streaming labels.

    Missing/non-scalar online predictions count as incorrect. This mirrors the
    default Phase-2 prequential protocol and avoids sklearn target-type coercion.
    """
    yt = np.asarray(list(y_true), dtype=object).reshape(-1)
    yp = np.asarray(list(y_pred), dtype=object).reshape(-1)
    if len(yt) != len(yp):
        raise ValueError("Prediction length mismatch: y_true=%s, y_pred=%s" % (len(yt), len(yp)))
    if len(yt) == 0:
        return 0.0
    correct = 0
    for truth, pred in zip(yt, yp):
        truth = _scalar(truth)
        pred = _scalar(pred)
        if pred is None or isinstance(pred, (dict, list, tuple, set, np.ndarray, pd.Series)):
            continue
        try:
            correct += int(bool(pred == truth))
        except Exception:
            continue
    return float(correct / len(yt))


def _unique_labels(y: Iterable[Any]) -> list[Any]:
    labels = []
    for value in list(y):
        value = _scalar(value)
        if not any(_labels_equal(value, old) for old in labels):
            labels.append(value)
    return labels


def _labels_equal(a: Any, b: Any) -> bool:
    try:
        return bool(_scalar(a) == _scalar(b))
    except Exception:
        return False


def _hard_probability_matrix(model, X: pd.DataFrame, classes: list[Any]) -> np.ndarray:
    pred = _safe_predict(model, X)
    out = np.zeros((len(pred), len(classes)), dtype=float)
    for i, value in enumerate(pred):
        for j, label in enumerate(classes):
            if _labels_equal(value, label):
                out[i, j] = 1.0
                break
    return out


def _safe_probability_matrix(model, X: pd.DataFrame, classes: list[Any]) -> tuple[Optional[np.ndarray], str]:
    """Return class-probability matrix plus provenance.

    Phase 3 prefers genuine probabilities. A hard-label proxy is intentionally
    *not* used for SHAP/LIME because it produces brittle explanations and can
    make a convincing plot from a non-probabilistic decision surface.
    """
    if len(classes) < 2:
        return None, "single_class"

    # Batch API (isolated workers override this for one IPC round trip).
    if hasattr(model, "predict_proba"):
        try:
            raw = model.predict_proba(X)
            if isinstance(raw, list) and raw and all(isinstance(r, dict) for r in raw):
                arr = np.asarray([[float(r.get(label, 0.0) or 0.0) for label in classes] for r in raw], dtype=float)
            else:
                arr = np.asarray(raw, dtype=float)
            if arr.ndim == 2 and arr.shape == (len(X), len(classes)):
                rowsum = arr.sum(axis=1, keepdims=True)
                rowsum[rowsum <= 0] = 1.0
                return np.clip(arr / rowsum, 0.0, 1.0), "predict_proba"
        except Exception:
            pass

    # River-style row API.
    if hasattr(model, "predict_proba_one"):
        matrix = []
        try:
            for row in X.to_dict(orient="records"):
                proba = model.predict_proba_one(row)
                if not isinstance(proba, dict) or not proba:
                    return None, "predict_proba_one_unavailable"
                matrix.append([float(proba.get(label, 0.0) or 0.0) for label in classes])
            arr = np.asarray(matrix, dtype=float)
            rowsum = arr.sum(axis=1, keepdims=True)
            rowsum[rowsum <= 0] = 1.0
            return np.clip(arr / rowsum, 0.0, 1.0), "predict_proba_one"
        except Exception:
            pass

    return None, "probabilities_unavailable"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def _hoyer_sparsity(v: np.ndarray) -> float:
    v = np.abs(np.asarray(v, dtype=float))
    n = len(v)
    if n <= 1 or np.linalg.norm(v) == 0:
        return 0.0
    return float((np.sqrt(n) - (v.sum() / np.linalg.norm(v))) / (np.sqrt(n) - 1))


def _perm_vector(model, X: pd.DataFrame, y: pd.Series, seed: int, repeats: int = 3) -> np.ndarray:
    base = _safe_accuracy(y, _safe_predict(model, X))
    rng = np.random.default_rng(seed)
    values = []
    for col in X.columns:
        drops = []
        original = X[col].to_numpy(copy=True)
        for _ in range(max(1, int(repeats))):
            xp = X.copy()
            xp[col] = rng.permutation(original)
            perm_acc = _safe_accuracy(y, _safe_predict(model, xp))
            drops.append(base - perm_acc)
        values.append(max(0.0, float(np.mean(drops))))
    return np.asarray(values, dtype=float)


def _shap_vector_from_values(values: Any, n_features: int) -> np.ndarray:
    """Aggregate SHAP outputs across samples/classes into one absolute vector."""
    if isinstance(values, list):
        arrays = [np.asarray(v, dtype=float) for v in values]
        arr = np.stack([np.mean(np.abs(v), axis=0) for v in arrays], axis=0)
        vec = np.mean(arr, axis=0)
    else:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:
            # SHAP versions differ between [samples, features, outputs] and
            # [outputs, samples, features]. Choose the axis matching features.
            if arr.shape[1] == n_features:
                vec = np.mean(np.abs(arr), axis=(0, 2))
            elif arr.shape[2] == n_features:
                vec = np.mean(np.abs(arr), axis=(0, 1))
            else:
                raise ValueError("Unable to locate SHAP feature axis.")
        elif arr.ndim == 2:
            vec = np.mean(np.abs(arr), axis=0)
        elif arr.ndim == 1:
            vec = np.abs(arr)
        else:
            raise ValueError("Unexpected SHAP output shape: %r" % (arr.shape,))
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if len(vec) != n_features:
        raise ValueError("SHAP feature length mismatch: %s != %s" % (len(vec), n_features))
    return vec


def _shap_vectors(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    background_rows: int = 12,
    explain_rows: int = 6,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if shap is None:
        raise RuntimeError("SHAP is not installed.")
    classes = _unique_labels(y)
    probe, provenance = _safe_probability_matrix(model, X.head(min(3, len(X))), classes)
    if probe is None:
        raise RuntimeError("SHAP requires class probabilities; %s." % provenance)

    rng = np.random.default_rng(seed)
    bg_n = min(max(4, int(background_rows)), len(X))
    bg_idx = rng.choice(len(X), size=bg_n, replace=False)
    background = X.iloc[np.sort(bg_idx)].copy()

    def predict_fn(values):
        frame = pd.DataFrame(values, columns=X.columns)
        probs, source = _safe_probability_matrix(model, frame, classes)
        if probs is None:
            raise RuntimeError("Probability API became unavailable during SHAP: %s" % source)
        return probs

    explainer = shap.KernelExplainer(predict_fn, background)
    vectors = []
    sample_n = min(max(2, int(explain_rows)), len(X))
    nsamples = max(20, min(80, 2 * X.shape[1] + 8))
    for rep in range(3):
        rr = np.random.default_rng(seed + 101 * rep)
        idx = rr.choice(len(X), size=sample_n, replace=False)
        sample = X.iloc[np.sort(idx)]
        values = explainer.shap_values(sample, nsamples=nsamples, silent=True)
        vectors.append(_shap_vector_from_values(values, X.shape[1]))
    return vectors, {"probability_source": provenance, "classes": [str(c) for c in classes], "nsamples": nsamples}


def _lime_vectors(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    num_samples: int = 300,
    categorical_features: Optional[Iterable[str]] = None,
    categorical_value_names: Optional[dict[str, dict[str, int]]] = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if LimeTabularExplainer is None:
        raise RuntimeError("LIME is not installed.")
    classes = _unique_labels(y)
    probe, provenance = _safe_probability_matrix(model, X.head(min(3, len(X))), classes)
    if probe is None:
        raise RuntimeError("LIME requires class probabilities; %s." % provenance)

    def predict_fn(values):
        frame = pd.DataFrame(values, columns=X.columns)
        probs, source = _safe_probability_matrix(model, frame, classes)
        if probs is None:
            raise RuntimeError("Probability API became unavailable during LIME: %s" % source)
        return probs

    categorical_set = {str(c) for c in (categorical_features or [])}
    categorical_idx = [i for i, c in enumerate(X.columns) if str(c) in categorical_set]
    categorical_names = {}
    encoded_name_maps = categorical_value_names or {}
    for idx in categorical_idx:
        col = str(X.columns[idx])
        # StreamingEncoder intentionally reserves code 0 for missing/unknown and
        # assigns observed categories from 1..K. LIME indexes categorical_names
        # directly with the encoded integer, so the name array must preserve
        # those integer positions instead of being a compact 0..K-1 list.
        mapping = encoded_name_maps.get(col) or {}
        if mapping:
            max_code = max([int(v) for v in mapping.values()] + [0])
            names = ["<missing/unknown>"] + ["<unseen>"] * max_code
            for raw_name, code in mapping.items():
                code_i = int(code)
                if code_i < 0:
                    continue
                while len(names) <= code_i:
                    names.append("<unseen>")
                names[code_i] = str(raw_name)
            categorical_names[idx] = names
        else:
            observed = [int(v) for v in pd.Series(X.iloc[:, idx]).dropna().unique().tolist() if float(v).is_integer() and float(v) >= 0]
            max_code = max(observed + [0])
            categorical_names[idx] = [str(i) for i in range(max_code + 1)]

    vectors = []
    for rep in range(3):
        explainer = LimeTabularExplainer(
            training_data=X.to_numpy(dtype=float),
            feature_names=[str(c) for c in X.columns],
            class_names=[str(c) for c in classes],
            categorical_features=categorical_idx or None,
            categorical_names=categorical_names or None,
            mode="classification",
            discretize_continuous=True,
            random_state=seed + rep * 17,
        )
        idx = int((seed + rep * 97) % len(X))
        exp = explainer.explain_instance(
            X.iloc[idx].to_numpy(dtype=float),
            predict_fn,
            num_features=X.shape[1],
            num_samples=max(100, int(num_samples)),
            top_labels=min(2, len(classes)),
        )
        vec = np.zeros(X.shape[1], dtype=float)
        mapping = exp.as_map()
        for _, pairs in mapping.items():
            for feature_idx, weight in pairs:
                if 0 <= int(feature_idx) < len(vec):
                    vec[int(feature_idx)] += abs(float(weight))
        vectors.append(vec)
    return vectors, {
        "probability_source": provenance,
        "classes": [str(c) for c in classes],
        "num_samples": int(num_samples),
        "categorical_feature_count": len(categorical_idx),
        "categorical_features": [str(X.columns[i]) for i in categorical_idx],
    }


def _quality_from_vectors(vectors: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Optional[float]]]:
    clean = [np.asarray(v, dtype=float).reshape(-1) for v in vectors]
    mean_v = np.mean(clean, axis=0)
    if not np.isfinite(mean_v).all() or float(np.abs(mean_v).sum()) <= 1e-12:
        raise ValueError("Explanation signal is degenerate (all-zero or non-finite importance).")

    k = max(1, min(5, len(mean_v)))
    top_sets = [set(np.argsort(np.abs(v))[::-1][:k].tolist()) for v in clean]
    jaccards = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = top_sets[i] | top_sets[j]
            jaccards.append(len(top_sets[i] & top_sets[j]) / len(union) if union else 1.0)

    cosines = [_cosine(clean[i], clean[j]) for i in range(len(clean)) for j in range(i + 1, len(clean))]
    sensitivities = []
    for i in range(len(clean)):
        for j in range(i + 1, len(clean)):
            a, b = clean[i], clean[j]
            den = np.abs(a).sum() + np.abs(b).sum()
            sensitivities.append(float(np.abs(a - b).sum() / den) if den > 0 else 0.0)

    return mean_v, {
        "stability": float(np.mean(jaccards)) if jaccards else None,
        "consistency": float(np.mean(cosines)) if cosines else None,
        "sensitivity": float(np.mean(sensitivities)) if sensitivities else None,
        "sparsity": _hoyer_sparsity(mean_v),
    }


def _method_order(preference: str) -> list[str]:
    pref = str(preference or "auto").strip().lower()
    if pref == "auto":
        return ["shap", "lime", "permutation"]
    if pref in {"shap", "lime", "permutation"}:
        return [pref]
    raise ValueError("xai method must be one of: auto, shap, lime, permutation")


def explain_framework(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
    max_rows: int = 250,
    method_preference: str = "auto",
    reference_accuracy: Optional[float] = None,
    categorical_features: Optional[Iterable[str]] = None,
    categorical_value_names: Optional[dict[str, dict[str, int]]] = None,
    replay_warning_threshold: float = 0.05,
) -> dict[str, Any]:
    """Journal-oriented streaming explanation cascade.

    Phase 3 attempts SHAP first, then LIME, then repeated permutation when
    ``method_preference='auto'``. SHAP/LIME are only accepted when the framework
    exposes genuine class probabilities. Every attempted method is recorded;
    degenerate all-zero explanations are rejected instead of being shown as a
    perfectly stable explanation.
    """
    if X is None or y is None or len(X) < 30 or X.shape[1] == 0:
        return {"status": "insufficient_data", "method": None, "method_attempts": []}

    try:
        X = X.tail(min(max_rows, len(X))).copy().reset_index(drop=True)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = pd.Series(y).tail(len(X)).reset_index(drop=True)
        attempts = []
        selected_method = None
        selected_meta = {}
        vectors = None

        for method in _method_order(method_preference):
            try:
                if method == "shap":
                    candidate, meta = _shap_vectors(model, X, y, seed)
                    label = "SHAP/Kernel"
                elif method == "lime":
                    candidate, meta = _lime_vectors(
                        model, X, y, seed,
                        categorical_features=categorical_features,
                        categorical_value_names=categorical_value_names,
                    )
                    label = "LIME/tabular"
                else:
                    candidate = [_perm_vector(model, X, y, seed + i * 17, repeats=2) for i in range(3)]
                    meta = {"metric": "accuracy_drop", "repeats_per_vector": 2}
                    label = "permutation/repeated"
                # Reject a method if its aggregate signal contains no information.
                _quality_from_vectors(candidate)
                attempts.append({"method": label, "status": "ok"})
                selected_method, selected_meta, vectors = label, meta, candidate
                break
            except Exception as exc:
                attempts.append({"method": method, "status": "failed_or_degenerate", "reason": "%s: %s" % (type(exc).__name__, exc)})

        if vectors is None or selected_method is None:
            return {
                "status": "unsupported",
                "method": None,
                "feature_importance": [],
                "method_attempts": attempts,
                "diagnostic_note": "No requested XAI method produced a non-degenerate explanation signal.",
            }

        mean_v, quality = _quality_from_vectors(vectors)
        total = float(np.abs(mean_v).sum())
        norm = np.abs(mean_v) / total
        importance = [
            {"feature": str(c), "importance": float(v)}
            for c, v in sorted(zip(X.columns, norm), key=lambda kv: kv[1], reverse=True)
        ]

        base_acc = _safe_accuracy(y, _safe_predict(model, X))
        k = max(1, min(5, X.shape[1]))
        xd = X.copy()
        for idx in np.argsort(np.abs(mean_v))[::-1][:k]:
            col = X.columns[idx]
            replacement = pd.to_numeric(X[col], errors="coerce").median()
            xd[col] = 0.0 if pd.isna(replacement) else float(replacement)
        degraded_acc = _safe_accuracy(y, _safe_predict(model, xd))
        fidelity = float(max(0.0, base_acc - degraded_acc))

        replay_gap = None
        replay_warning = None
        if reference_accuracy is not None:
            replay_gap = abs(float(reference_accuracy) - float(base_acc))
            threshold = max(0.0, float(replay_warning_threshold))
            if replay_gap >= threshold:
                replay_warning = (
                    "Current-model replay accuracy differs materially from the prequential rolling accuracy. "
                    "Interpret this explanation as final-state model behavior, not as a reconstruction of historical decisions."
                )

        return {
            "status": "ok",
            "method": selected_method,
            "feature_importance": importance,
            "stability": quality.get("stability"),
            "fidelity": fidelity,
            "sensitivity": quality.get("sensitivity"),
            "consistency": quality.get("consistency"),
            "sparsity": quality.get("sparsity"),
            "base_accuracy_on_explanation_window": float(base_acc),
            "prequential_reference_accuracy": None if reference_accuracy is None else float(reference_accuracy),
            "replay_accuracy_gap": replay_gap,
            "replay_warning_threshold": max(0.0, float(replay_warning_threshold)),
            "replay_warning": replay_warning,
            "deletion_accuracy": float(degraded_acc),
            "method_attempts": attempts,
            "method_metadata": selected_meta,
            "diagnostic_note": (
                "Explanation quality measures are perturbation/resampling diagnostics, not causal guarantees. "
                "SHAP/LIME are used only with genuine class-probability APIs; permutation is the model-agnostic fallback."
            ),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "method": None,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "diagnostic_note": "Framework run completed; explanation generation failed.",
        }
