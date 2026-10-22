from __future__ import annotations

"""Run-Studio framework, algorithm and hyperparameter controls.

These controls configure the *next* benchmark run. They never mutate an already
recorded result, and they expose only options supported by the current adapters.

v6.1 deliberately avoids nested ``st.expander`` calls because Streamlit rejects
expanders inside expanders. Run Studio may wrap this whole renderer in one outer
expander; per-algorithm override areas therefore use bordered containers + toggles.
"""

from copy import deepcopy
from typing import Any, Optional

import streamlit as st

from awareml.frameworks.catalog import (
    framework_catalog,
    default_framework_config,
    runtime_available_algorithms,
)


def _widget_value(label: str, spec: dict, value: Any, key: str):
    kind = str(spec.get("type") or "float")
    if kind == "bool":
        return bool(st.toggle(label, value=bool(value), key=key))
    if kind == "choice":
        choices = list(spec.get("choices") or [])
        if not choices:
            return value
        default = value if value in choices else spec.get("default", choices[0])
        return st.selectbox(label, choices, index=choices.index(default), key=key)
    if kind == "int":
        return int(st.number_input(
            label,
            min_value=int(spec.get("min", 0)),
            max_value=int(spec.get("max", 10**9)),
            value=int(value if value is not None else spec.get("default", 0)),
            step=max(1, int(spec.get("step", 1))),
            key=key,
        ))
    return float(st.number_input(
        label,
        min_value=float(spec.get("min", -1e12)),
        max_value=float(spec.get("max", 1e12)),
        value=float(value if value is not None else spec.get("default", 0.0)),
        step=float(spec.get("step", 0.01)),
        format="%.8g",
        key=key,
    ))


def _coerce_existing(framework: str, state: Optional[dict]) -> dict:
    default = default_framework_config(framework)
    stored = ((state or {}).get("framework_config_overrides") or {}).get(framework)
    if not isinstance(stored, dict):
        return default
    out = deepcopy(default)
    if isinstance(stored.get("algorithms"), list):
        out["algorithms"] = list(stored["algorithms"])
    out["global"].update(dict(stored.get("global") or {}))
    out["hyperparameters"] = deepcopy(stored.get("hyperparameters") or {})
    return out


def _single_algorithm_selector(
    label: str,
    available: list[str],
    requested: list[str],
    key: str,
) -> list[str]:
    if not available:
        return []
    default_algo = next((a for a in requested if a in available), available[0])
    chosen = st.selectbox(
        label,
        available,
        index=available.index(default_algo),
        key=key,
    )
    return [chosen]


def render_framework_configuration(
    selected_frameworks,
    seed: int = 42,
    key_prefix: str = "fwcfg",
    state: Optional[dict] = None,
    oaml_mode: Optional[str] = None,  # compatibility with v6 callers; v6.1 renders mode here
    persist: bool = True,
) -> dict:
    """Render framework configuration and return a framework->config mapping."""
    catalog = framework_catalog()
    selected = list(selected_frameworks or [])
    result = {}
    if not selected:
        return result

    st.markdown("**Framework algorithms & hyperparameters**")
    st.caption(
        "Configure algorithms and hyperparameters for the next benchmark run. "
        "Only backend-supported options are shown."
    )

    tabs = st.tabs(selected)
    for tab, framework in zip(tabs, selected):
        with tab:
            spec = catalog.get(framework) or {}
            current = _coerce_existing(framework, state)
            st.caption(spec.get("description") or "")
            available = runtime_available_algorithms(framework)
            requested = [a for a in current.get("algorithms", []) if a in available]
            if not requested:
                requested = [a for a in spec.get("default_algorithms", []) if a in available]

            # ----------------------------------------------------------
            # Algorithm selection: shown for every framework.
            # ----------------------------------------------------------
            if framework in {"AutoStreamML", "AutoClass"}:
                algorithms = st.multiselect(
                    "Algorithms",
                    available,
                    default=requested,
                    key=f"{key_prefix}_{framework}_algorithms",
                    help=(
                        "Adaptive pool can search/select among these backend-supported learners. "
                        "Fixed algorithm uses the first selected learner."
                    ),
                )
                if not algorithms:
                    algorithms = requested or available[:1]

            elif framework == "OAML":
                # OAML execution mode belongs with the OAML framework rather than
                # as a separate Run-Studio control above the framework selector.
                mode_choices = ["online", "gama"]
                stored_mode = str((current.get("global") or {}).get("mode") or oaml_mode or "online").lower()
                if stored_mode not in mode_choices:
                    stored_mode = "online"
                mode = st.selectbox(
                    "Execution mode",
                    mode_choices,
                    index=mode_choices.index(stored_mode),
                    key=f"{key_prefix}_{framework}_mode",
                    help="online uses the isolated River learner; gama uses the GAMA AutoPipeline bridge with River fallback.",
                )
                if mode == "gama":
                    algorithms = _single_algorithm_selector(
                        "Algorithm",
                        [a for a in available if a == "GAMA AutoPipeline"] or ["GAMA AutoPipeline"],
                        requested,
                        f"{key_prefix}_{framework}_algorithm",
                    )
                else:
                    algorithms = _single_algorithm_selector(
                        "Algorithm",
                        [a for a in available if a != "GAMA AutoPipeline"],
                        requested,
                        f"{key_prefix}_{framework}_algorithm",
                    )

            else:
                # EvoAutoML and ChaCha currently expose one trustworthy backend
                # algorithm each. A real selector is still shown so every
                # framework presents algorithm provenance consistently.
                algorithms = _single_algorithm_selector(
                    "Algorithm",
                    available or list(spec.get("default_algorithms") or []),
                    requested,
                    f"{key_prefix}_{framework}_algorithm",
                )

            # ----------------------------------------------------------
            # Framework-level controls.
            # ----------------------------------------------------------
            globals_out = {}
            global_specs = spec.get("global") or {}
            if global_specs:
                st.markdown("**Framework-level controls**")
                cols = st.columns(2)
                display_specs = [
                    (name, pspec)
                    for name, pspec in global_specs.items()
                    if not (framework == "OAML" and name == "mode")
                ]
                for i, (name, pspec) in enumerate(display_specs):
                    with cols[i % 2]:
                        globals_out[name] = _widget_value(
                            name.replace("_", " ").title(),
                            pspec,
                            (current.get("global") or {}).get(name, pspec.get("default")),
                            f"{key_prefix}_{framework}_global_{name}",
                        )
                if framework == "OAML":
                    globals_out["mode"] = mode

            # ----------------------------------------------------------
            # Exact per-algorithm overrides WITHOUT nested expanders.
            # ----------------------------------------------------------
            hp_out = {}
            algorithm_specs = spec.get("algorithms") or {}
            for algorithm in algorithms:
                params = (algorithm_specs.get(algorithm) or {}).get("parameters") or {}
                if not params:
                    continue
                existing = (current.get("hyperparameters") or {}).get(algorithm) or {}
                with st.container(border=True):
                    st.markdown("**{} · hyperparameter overrides**".format(algorithm))
                    show_overrides = st.toggle(
                        "Customize exact hyperparameters",
                        value=bool(existing),
                        key=f"{key_prefix}_{framework}_{algorithm}_show_overrides",
                        help=(
                            "Turn this on to lock selected parameters to exact values. "
                            "Leave parameters unlocked to keep framework/default search behavior."
                        ),
                    )
                    overrides = {}
                    if show_overrides:
                        st.caption(
                            "Only enabled overrides are locked. Other parameters keep the framework/default behavior."
                        )
                        for pname, pspec in params.items():
                            enabled = st.checkbox(
                                "Override {}".format(pname),
                                value=pname in existing,
                                key=f"{key_prefix}_{framework}_{algorithm}_{pname}_enabled",
                            )
                            if enabled:
                                overrides[pname] = _widget_value(
                                    pname.replace("_", " ").title(),
                                    pspec,
                                    existing.get(pname, pspec.get("default")),
                                    f"{key_prefix}_{framework}_{algorithm}_{pname}_value",
                                )
                    if overrides:
                        hp_out[algorithm] = overrides

            result[framework] = {
                "algorithms": list(algorithms),
                "global": globals_out,
                "hyperparameters": hp_out,
                "seed": int(seed),
            }

    if persist and state is not None:
        state["framework_config_overrides"] = deepcopy(result)
    return result
