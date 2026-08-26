from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


TARGET_NAME_TOKENS = {
    "target", "label", "class", "outcome", "response", "y",
    "diagnosis", "default", "churn", "fraud", "approved", "income",
    "occupation_binary", "survived", "status", "category",
}

SENSITIVE_NAME_TOKENS = {
    "sex", "gender", "race", "ethnicity", "age", "religion",
    "marital", "marital_status", "disability", "nationality",
    "citizenship", "group", "protected", "sensitive",
}

ORDER_NAME_TOKENS = {
    "time", "timestamp", "date", "datetime", "year", "month",
    "day", "index", "time_index", "sequence", "order",
}

ID_NAME_TOKENS = {
    "id", "uuid", "identifier", "record_id", "row_id", "customer_id",
    "user_id", "case_id", "transaction_id",
}


def _name_score(name: str, tokens) -> int:
    n = str(name).strip().lower()
    parts = set(n.replace("-", "_").split("_"))
    if n in tokens:
        return 100
    return 30 * sum(1 for token in tokens if token in parts or token in n)


def _is_probable_identifier(series: pd.Series, name: str) -> bool:
    n = len(series)
    if n == 0:
        return False
    unique = int(series.nunique(dropna=True))
    name_score = _name_score(name, ID_NAME_TOKENS)
    return unique >= max(20, int(0.95 * n)) and (name_score > 0 or unique == n)


def suggest_targets(df: pd.DataFrame, limit: int = 5) -> List[Dict[str, object]]:
    n = max(1, len(df))
    candidates = []
    for col in df.columns:
        s = df[col]
        unique = int(s.nunique(dropna=True))
        if unique < 2:
            continue
        if _is_probable_identifier(s, str(col)):
            continue
        if _name_score(str(col), ORDER_NAME_TOKENS) > 0 and unique > 20:
            continue

        name_score = _name_score(str(col), TARGET_NAME_TOKENS)
        class_like = unique <= max(20, int(np.sqrt(n)))
        categorical = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_categorical_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        )

        score = name_score
        if class_like:
            score += 25
        if categorical:
            score += 10
        if 2 <= unique <= 10:
            score += 10
        if unique > max(100, int(0.20 * n)):
            score -= 30

        if score > 0:
            candidates.append({
                "column": str(col),
                "score": int(score),
                "classes": unique,
                "reason": (
                    "target-like name" if name_score > 0 else "classification-like cardinality"
                ),
            })

    candidates.sort(key=lambda x: (-int(x["score"]), int(x["classes"]), str(x["column"])))
    return candidates[:limit]


def suggest_sensitive_attributes(df: pd.DataFrame, target: Optional[str] = None) -> List[Dict[str, object]]:
    suggestions = []
    for col in df.columns:
        if target is not None and str(col) == str(target):
            continue
        score = _name_score(str(col), SENSITIVE_NAME_TOKENS)
        if score <= 0:
            continue
        unique = int(df[col].nunique(dropna=True))
        suggestions.append({
            "column": str(col),
            "groups": unique,
            "reason": "name suggests a possible fairness-audit attribute",
        })
    suggestions.sort(key=lambda x: (int(x["groups"]), str(x["column"])))
    return suggestions


def suggest_order_columns(df: pd.DataFrame) -> List[str]:
    rows = []
    for col in df.columns:
        if _name_score(str(col), ORDER_NAME_TOKENS) > 0:
            rows.append(str(col))
    return rows[:5]


def leakage_warnings(df: pd.DataFrame, target: Optional[str]) -> List[str]:
    if not target or target not in df.columns:
        return []

    warnings = []
    y = df[target]
    y_unique = int(y.nunique(dropna=True))
    if y_unique < 2:
        warnings.append("The selected target is constant; classification is not meaningful.")
        return warnings

    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        unique = int(s.nunique(dropna=True))
        if unique < 2:
            continue

        if _is_probable_identifier(s, str(col)):
            warnings.append(
                "`{}` looks like a near-unique identifier. Usually exclude identifiers from model features.".format(col)
            )
            continue

        # Detect a feature whose values deterministically encode the selected target.
        # Limit cardinality to avoid flagging a unique identifier as deterministic leakage.
        if unique <= min(200, max(20, int(0.10 * len(df)))):
            valid = pd.DataFrame({"x": s, "y": y}).dropna()
            if not valid.empty:
                mapping = valid.groupby("x", dropna=False)["y"].nunique(dropna=True)
                if len(mapping) >= 2 and int(mapping.max()) == 1:
                    warnings.append(
                        "Potential target leakage: `{}` deterministically maps to `{}` in the current data. "
                        "Review or exclude it before modeling.".format(col, target)
                    )

    return warnings


def dataset_advice(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, object]:
    if df is None:
        return {}

    missing_fraction = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    numeric = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
    categorical = [str(c) for c in df.columns if str(c) not in numeric]
    current_target = str(target) if target in df.columns else None

    target_candidates = suggest_targets(df)
    sensitive = suggest_sensitive_attributes(df, current_target)
    order_cols = suggest_order_columns(df)
    warnings = leakage_warnings(df, current_target)

    if current_target:
        n_classes = int(df[current_target].nunique(dropna=True))
    else:
        n_classes = None

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "missing_fraction": missing_fraction,
        "current_target": current_target,
        "n_classes": n_classes,
        "target_candidates": target_candidates,
        "sensitive_candidates": sensitive,
        "order_candidates": order_cols,
        "warnings": warnings,
    }


def render_dataset_advisor(df: Optional[pd.DataFrame], dataset_name: Optional[str], target: Optional[str]) -> None:
    if df is None:
        st.info(
            "Upload or select a dataset below. After it is loaded, AwareML will show a dataset-specific "
            "setup assistant here with target candidates, possible fairness-audit attributes, stream-order "
            "candidates, and leakage warnings. Suggestions are advisory only."
        )
        return

    advice = dataset_advice(df, target)
    title = "Dataset setup assistant · {}".format(dataset_name or "active dataset")

    with st.expander(title, expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", "{:,}".format(advice["rows"]))
        with c2:
            st.metric("Columns", str(advice["columns"]))
        with c3:
            st.metric("Numeric", str(advice["numeric_features"]))
        with c4:
            st.metric("Missing", "{:.2%}".format(advice["missing_fraction"]))

        if advice["current_target"]:
            st.success(
                "Current target: `{}` · observed classes: `{}`".format(
                    advice["current_target"], advice["n_classes"]
                )
            )
        else:
            candidates = advice["target_candidates"]
            if candidates:
                st.info(
                    "Possible target columns: {}. Confirm the target manually in Run Studio.".format(
                        ", ".join("`{}`".format(item["column"]) for item in candidates)
                    )
                )
            else:
                st.info(
                    "No confident target candidate was detected from the schema. Select the target manually."
                )

        sensitive = advice["sensitive_candidates"]
        if sensitive:
            st.warning(
                "Possible fairness-audit attributes: {}. These are suggestions from column names only; "
                "AwareML does not auto-enable a sensitive attribute. Confirm using domain knowledge.".format(
                    ", ".join("`{}`".format(item["column"]) for item in sensitive)
                )
            )
        else:
            st.caption(
                "No obvious fairness-audit attribute was detected from column names. Leave it unset unless "
                "domain knowledge identifies a protected/sensitive attribute."
            )

        order_cols = advice["order_candidates"]
        if order_cols:
            st.info(
                "Possible stream-order/time columns: {}. If none is explicitly used by the loader, CSV row "
                "order remains the processing order.".format(
                    ", ".join("`{}`".format(col) for col in order_cols)
                )
            )
        else:
            st.caption(
                "No explicit time/order column was detected. AwareML will process the existing row order as "
                "stream order; do not automatically interpret that order as real chronology."
            )

        if advice["warnings"]:
            st.markdown("**Data-quality / leakage checks**")
            for warning in advice["warnings"]:
                st.error(warning)
        else:
            st.success("No obvious schema-level leakage or identifier warning was detected by the quick advisor.")

        st.caption(
            "Advisor scope: schema/cardinality/name-based checks only. It does not replace domain validation "
            "or automatically change your dataset, target, sensitive attribute, or features."
        )
