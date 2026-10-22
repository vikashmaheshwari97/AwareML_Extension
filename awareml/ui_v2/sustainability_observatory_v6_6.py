from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plots import FRAMEWORK_COLORS, apply_research_layout


RESOURCE_COLUMNS = {
    "Runtime": ("Runtime s", "s"),
    "Energy": ("Energy kWh", "kWh"),
    "CO₂": ("CO₂ kg", "kg"),
}

FRAMEWORK_ORDER = ["AutoStreamML", "AutoClass", "EvoAutoML", "OAML", "ChaCha"]


def sustainability_frame(results: Sequence[dict]) -> pd.DataFrame:
    rows: List[dict] = []
    for result in results:
        s = result.get("sustainability") or {}
        rows.append(
            {
                "Framework": result.get("framework"),
                "Status": s.get("status"),
                "Accuracy": result.get("accuracy"),
                "Macro-F1": result.get("f1_macro"),
                "Runtime s": result.get("runtime_sec"),
                "Energy kWh": result.get("energy_kwh"),
                "CO₂ kg": result.get("co2_kg"),
                "Samples": result.get("samples"),
                "Measurement duration s": s.get("duration_sec"),
                "Backend": s.get("measurement_backend"),
                "CodeCarbon": s.get("codecarbon_version"),
                "CPU": s.get("cpu"),
                "GPU": s.get("gpu"),
                "RAM GB": s.get("ram_gb"),
                "Physical CPUs": s.get("physical_cpus"),
                "Country": s.get("country_iso"),
                "Region": s.get("region"),
                "Carbon intensity gCO₂/kWh": s.get("carbon_intensity_g_per_kwh"),
                "Carbon intensity source": s.get("carbon_intensity_source"),
                "Warm-up s": s.get("warmup_sec"),
                "Warm-up samples": s.get("warmup_samples"),
                "Repetition": s.get("repetition_id"),
                "Repetitions planned": s.get("repetitions_planned"),
                "Failure reason": s.get("measurement_failure_reason"),
            }
        )
    frame = pd.DataFrame(rows)
    for col in [
        "Accuracy",
        "Macro-F1",
        "Runtime s",
        "Energy kWh",
        "CO₂ kg",
        "Samples",
        "Measurement duration s",
        "Carbon intensity gCO₂/kWh",
        "RAM GB",
    ]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if not frame.empty:
        frame["Framework"] = frame["Framework"].astype(str)
        order_map = {name: i for i, name in enumerate(FRAMEWORK_ORDER)}
        frame["_order"] = frame["Framework"].map(order_map).fillna(999)
        frame = frame.sort_values(["_order", "Framework"]).drop(columns=["_order"]).reset_index(drop=True)
    return frame


def _safe_min(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min())


def _safe_max(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.max())


def _fmt_value(value: float, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    if unit == "s":
        return f"{value:.2f} s"
    if unit == "kWh":
        if abs(value) < 0.01:
            return f"{value:.3e} kWh"
        return f"{value:.4f} kWh"
    if unit == "kg":
        if abs(value) < 0.01:
            return f"{value:.3e} kg"
        return f"{value:.4f} kg"
    return f"{value:.4g}"


def _pareto_mask(frame: pd.DataFrame, resource_col: str) -> np.ndarray:
    data = frame[["Accuracy", resource_col]].copy()
    acc = pd.to_numeric(data["Accuracy"], errors="coerce").to_numpy(dtype=float)
    res = pd.to_numeric(data[resource_col], errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(data), dtype=bool)
    for i in range(len(data)):
        if not np.isfinite(acc[i]) or not np.isfinite(res[i]):
            continue
        dominated = False
        for j in range(len(data)):
            if i == j or not np.isfinite(acc[j]) or not np.isfinite(res[j]):
                continue
            at_least_as_good = acc[j] >= acc[i] and res[j] <= res[i]
            strictly_better = acc[j] > acc[i] or res[j] < res[i]
            if at_least_as_good and strictly_better:
                dominated = True
                break
        mask[i] = not dominated
    return mask


def resource_signature_figure(frame: pd.DataFrame) -> go.Figure:
    """Three aligned Cleveland/lollipop panels with exact resource values."""
    data = frame.copy()
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.075,
        subplot_titles=("Runtime ↓", "Energy ↓", "CO₂ ↓"),
    )

    for col_idx, (title, (metric, unit)) in enumerate(RESOURCE_COLUMNS.items(), start=1):
        values = pd.to_numeric(data[metric], errors="coerce")
        best = _safe_min(values)
        finite = values.dropna()
        xmax = float(finite.max()) if not finite.empty else 1.0
        xmax = max(xmax * 1.16, 1e-12)

        for row_idx, row in data.iterrows():
            fw = str(row["Framework"])
            value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            color = FRAMEWORK_COLORS.get(fw, "#64748b")
            ratio = (float(value) / best) if best and best > 0 else np.nan
            is_best = bool(best is not None and abs(float(value) - best) <= max(abs(best) * 1e-9, 1e-15))

            fig.add_trace(
                go.Scatter(
                    x=[0.0, float(value)],
                    y=[fw, fw],
                    mode="lines",
                    line={"color": "rgba(148,163,184,0.48)", "width": 3},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=[float(value)],
                    y=[fw],
                    mode="markers+text",
                    marker={
                        "size": 14 if is_best else 11,
                        "symbol": "diamond" if is_best else "circle",
                        "color": color,
                        "line": {"width": 2 if is_best else 0.8, "color": "#0f172a"},
                    },
                    text=[_fmt_value(float(value), unit)],
                    textposition="middle right",
                    textfont={"size": 9},
                    customdata=[[ratio]],
                    hovertemplate=(
                        f"<b>{fw}</b><br>{title}: {_fmt_value(float(value), unit)}"
                        "<br>Relative to best: %{customdata[0]:.2f}×<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=col_idx,
            )

        fig.update_xaxes(
            range=[0, xmax],
            title_text=f"{title} ({unit})",
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(148,163,184,0.16)",
            row=1,
            col=col_idx,
        )

    fig.update_yaxes(title_text=None, autorange="reversed", row=1, col=1)
    apply_research_layout(
        fig,
        height=450,
        legend="none",
        title="Measured resource signature · exact values · diamond = run minimum",
        bottom_margin=58,
    )
    fig.update_layout(
        margin={"l": 118, "r": 42, "t": 72, "b": 62},
        hovermode="closest",
    )
    return fig


def efficiency_ratio_figure(frame: pd.DataFrame) -> go.Figure:
    """Transparent ratio-to-best comparison; 1× is best for each resource."""
    data = frame.copy()
    metrics = [
        ("Runtime s", "Runtime"),
        ("Energy kWh", "Energy"),
        ("CO₂ kg", "CO₂"),
    ]
    fig = go.Figure()
    for metric, label in metrics:
        values = pd.to_numeric(data[metric], errors="coerce")
        best = _safe_min(values)
        if best is None or best <= 0:
            continue
        ratios = values / best
        fig.add_trace(
            go.Scatter(
                x=ratios,
                y=data["Framework"],
                mode="markers+text",
                name=label,
                text=[f"{v:.2f}×" if pd.notna(v) else "" for v in ratios],
                textposition="top center",
                marker={"size": 11},
                customdata=np.column_stack([values.to_numpy(dtype=float)]),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + label
                    + " relative to run minimum: %{x:.2f}×"
                    + "<br>Exact value: %{customdata[0]:.6g}<extra></extra>"
                ),
            )
        )
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_width=2,
        line_color="rgba(15,23,42,0.35)",
        annotation_text="run minimum = 1.0×",
        annotation_position="top",
    )
    fig.update_yaxes(autorange="reversed")
    apply_research_layout(
        fig,
        height=390,
        legend="bottom",
        title="Resource-cost ratios · how many times the best observed resource use?",
        bottom_margin=92,
    )
    fig.update_layout(
        xaxis={
            "title": "Relative resource cost (× run minimum) ↓",
            "rangemode": "tozero",
            "gridcolor": "rgba(148,163,184,0.16)",
        },
        margin={"l": 110, "r": 28, "t": 62, "b": 92},
    )
    return fig


def tradeoff_atlas_figure(frame: pd.DataFrame) -> go.Figure:
    """Accuracy versus Runtime, Energy and CO2 with a Pareto skyline in every panel."""
    data = frame.copy()
    specs = [
        ("Runtime s", "Runtime (s) ↓"),
        ("Energy kWh", "Energy (kWh) ↓"),
        ("CO₂ kg", "CO₂ (kg) ↓"),
    ]
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.075,
        subplot_titles=("Accuracy ↔ Runtime", "Accuracy ↔ Energy", "Accuracy ↔ CO₂"),
    )

    for idx, (resource_col, x_title) in enumerate(specs, start=1):
        mask = _pareto_mask(data, resource_col)
        pareto = data.loc[mask].dropna(subset=["Accuracy", resource_col]).sort_values(resource_col)

        if len(pareto) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=pareto[resource_col],
                    y=pareto["Accuracy"],
                    mode="lines",
                    line={"dash": "dash", "width": 2.5, "color": "#0f766e"},
                    name="Pareto frontier" if idx == 1 else None,
                    showlegend=(idx == 1),
                    hoverinfo="skip",
                ),
                row=1,
                col=idx,
            )

        for row_pos, row in data.iterrows():
            fw = str(row["Framework"])
            x = row.get(resource_col)
            y = row.get("Accuracy")
            if pd.isna(x) or pd.isna(y):
                continue
            is_pareto = bool(mask[row_pos])
            color = FRAMEWORK_COLORS.get(fw, "#64748b")
            fig.add_trace(
                go.Scatter(
                    x=[float(x)],
                    y=[float(y)],
                    mode="markers+text",
                    text=[fw],
                    textposition="top center",
                    textfont={"size": 9},
                    marker={
                        "size": 14 if is_pareto else 10,
                        "symbol": "diamond" if is_pareto else "circle",
                        "color": color,
                        "line": {"width": 2.2 if is_pareto else 0.8, "color": "#0f172a"},
                    },
                    showlegend=False,
                    customdata=[[resource_col]],
                    hovertemplate=(
                        f"<b>{fw}</b><br>Accuracy: %{{y:.4f}}"
                        f"<br>{x_title.replace(' ↓','')}: %{{x:.6g}}"
                        f"<br>Pareto efficient: {'Yes' if is_pareto else 'No'}<extra></extra>"
                    ),
                ),
                row=1,
                col=idx,
            )
        fig.update_xaxes(
            title_text=x_title,
            gridcolor="rgba(148,163,184,0.16)",
            row=1,
            col=idx,
        )
        fig.update_yaxes(
            title_text="Prequential accuracy ↑" if idx == 1 else None,
            gridcolor="rgba(148,163,184,0.16)",
            row=1,
            col=idx,
        )

    apply_research_layout(
        fig,
        height=470,
        legend="bottom",
        title="Accuracy–resource trade-off atlas · diamonds are nondominated",
        bottom_margin=86,
    )
    fig.update_layout(margin={"l": 62, "r": 28, "t": 72, "b": 86})
    return fig


def accuracy_cost_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Auditable exact trade-off table with transparent ratios."""
    data = frame.copy()
    acc_best = _safe_max(data["Accuracy"])
    for metric, label in [
        ("Runtime s", "Runtime × best"),
        ("Energy kWh", "Energy × best"),
        ("CO₂ kg", "CO₂ × best"),
    ]:
        best = _safe_min(data[metric])
        if best is not None and best > 0:
            data[label] = pd.to_numeric(data[metric], errors="coerce") / best
        else:
            data[label] = np.nan
    if acc_best is not None:
        data["Accuracy gap to best (pp)"] = (
            (acc_best - pd.to_numeric(data["Accuracy"], errors="coerce")) * 100.0
        )
    else:
        data["Accuracy gap to best (pp)"] = np.nan
    cols = [
        "Framework",
        "Accuracy",
        "Accuracy gap to best (pp)",
        "Runtime s",
        "Runtime × best",
        "Energy kWh",
        "Energy × best",
        "CO₂ kg",
        "CO₂ × best",
    ]
    return data[[c for c in cols if c in data.columns]]


def carbon_accounting_figure(frame: pd.DataFrame) -> go.Figure:
    """Energy-to-CO2 relation plus contribution shares, without redundant intensity dots."""
    data = frame.dropna(subset=["Energy kWh", "CO₂ kg"]).copy()
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Energy → CO₂ accounting relation",
            "Experiment contribution shares",
        ),
    )
    if data.empty:
        return fig

    energy = pd.to_numeric(data["Energy kWh"], errors="coerce")
    co2 = pd.to_numeric(data["CO₂ kg"], errors="coerce")
    valid = (energy > 0) & co2.notna()
    intensity = (co2[valid] / energy[valid]) * 1000.0
    median_intensity = float(intensity.median()) if not intensity.empty else np.nan

    for _, row in data.iterrows():
        fw = str(row["Framework"])
        x = float(row["Energy kWh"])
        y = float(row["CO₂ kg"])
        color = FRAMEWORK_COLORS.get(fw, "#64748b")
        obs_int = (y / x) * 1000.0 if x > 0 else np.nan
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[fw],
                textposition="top center",
                textfont={"size": 9},
                marker={"size": 12, "color": color, "line": {"width": 1.1, "color": "#0f172a"}},
                customdata=[[obs_int]],
                hovertemplate=(
                    f"<b>{fw}</b><br>Energy: %{{x:.6g}} kWh"
                    "<br>CO₂: %{y:.6g} kg"
                    "<br>Observed intensity: %{customdata[0]:.1f} gCO₂/kWh<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if np.isfinite(median_intensity):
        xmin = 0.0
        xmax = max(float(energy.max()) * 1.08, 1e-12)
        xs = np.array([xmin, xmax])
        ys = xs * median_intensity / 1000.0
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"dash": "dash", "width": 2, "color": "#475569"},
                name=f"Median intensity {median_intensity:.1f} gCO₂/kWh",
                hoverinfo="skip",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    totals = {
        "Energy": float(energy.sum()),
        "CO₂": float(co2.sum()),
    }
    for metric_name, metric_col, total in [
        ("Energy", "Energy kWh", totals["Energy"]),
        ("CO₂", "CO₂ kg", totals["CO₂"]),
    ]:
        if total <= 0:
            continue
        for _, row in data.iterrows():
            fw = str(row["Framework"])
            value = float(row[metric_col])
            share = 100.0 * value / total
            fig.add_trace(
                go.Bar(
                    x=[share],
                    y=[metric_name],
                    orientation="h",
                    name=fw,
                    legendgroup=fw,
                    marker={"color": FRAMEWORK_COLORS.get(fw, "#64748b")},
                    showlegend=(metric_name == "Energy"),
                    customdata=[[value]],
                    hovertemplate=(
                        f"<b>{fw}</b><br>{metric_name} share: %{{x:.1f}}%"
                        "<br>Exact contribution: %{customdata[0]:.6g}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

    fig.update_layout(barmode="stack")
    fig.update_xaxes(
        title_text="Measured energy (kWh) ↓",
        gridcolor="rgba(148,163,184,0.16)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Measured CO₂ (kg) ↓",
        gridcolor="rgba(148,163,184,0.16)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Share of experiment total (%)",
        range=[0, 100],
        gridcolor="rgba(148,163,184,0.16)",
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text=None, row=1, col=2)

    apply_research_layout(
        fig,
        height=460,
        legend="bottom",
        title="Carbon accounting · conversion consistency and contribution structure",
        bottom_margin=104,
    )
    fig.update_layout(margin={"l": 66, "r": 28, "t": 72, "b": 104})
    return fig


def extract_resource_timeline(results: Sequence[dict]) -> pd.DataFrame:
    """Best-effort reader for future/optional per-window sustainability telemetry.

    No synthetic timeline is created. The function returns an empty frame when the
    benchmark only contains final CodeCarbon totals.
    """
    rows: List[dict] = []
    for result in results:
        fw = str(result.get("framework") or "")
        sustainability = result.get("sustainability") or {}
        candidates = [
            sustainability.get("timeline"),
            sustainability.get("telemetry"),
            sustainability.get("resource_timeline"),
            result.get("resource_timeline"),
        ]
        series = next((item for item in candidates if isinstance(item, list) and item), None)
        if not series:
            continue
        for point in series:
            if not isinstance(point, dict):
                continue
            x = (
                point.get("sample")
                if point.get("sample") is not None
                else point.get("window")
                if point.get("window") is not None
                else point.get("elapsed_sec")
                if point.get("elapsed_sec") is not None
                else point.get("timestamp")
            )
            energy = point.get("energy_kwh")
            co2 = point.get("co2_kg")
            if x is None or (energy is None and co2 is None):
                continue
            rows.append(
                {
                    "Framework": fw,
                    "Step": x,
                    "Energy kWh": energy,
                    "CO₂ kg": co2,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["Energy kWh"] = pd.to_numeric(frame["Energy kWh"], errors="coerce")
        frame["CO₂ kg"] = pd.to_numeric(frame["CO₂ kg"], errors="coerce")
    return frame


def resource_timeline_figure(timeline: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Energy telemetry", "CO₂ telemetry"),
    )
    for fw, group in timeline.groupby("Framework", sort=False):
        color = FRAMEWORK_COLORS.get(str(fw), "#64748b")
        if group["Energy kWh"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=group["Step"],
                    y=group["Energy kWh"],
                    mode="lines+markers",
                    name=str(fw),
                    legendgroup=str(fw),
                    line={"color": color, "width": 2.4},
                    marker={"size": 5},
                    hovertemplate=f"<b>{fw}</b><br>Step %{{x}}<br>Energy %{{y:.6g}} kWh<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if group["CO₂ kg"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=group["Step"],
                    y=group["CO₂ kg"],
                    mode="lines+markers",
                    name=str(fw),
                    legendgroup=str(fw),
                    showlegend=False,
                    line={"color": color, "width": 2.4},
                    marker={"size": 5},
                    hovertemplate=f"<b>{fw}</b><br>Step %{{x}}<br>CO₂ %{{y:.6g}} kg<extra></extra>",
                ),
                row=2,
                col=1,
            )
    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="CO₂ (kg)", row=2, col=1)
    fig.update_xaxes(title_text="Measured step / sample", row=2, col=1)
    apply_research_layout(
        fig,
        height=520,
        legend="bottom",
        title="Measured resource timeline · only when real window-level telemetry exists",
        bottom_margin=94,
    )
    return fig


def sustainability_leaders(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    leaders: Dict[str, Optional[str]] = {}
    specs = {
        "accuracy": ("Accuracy", "max"),
        "runtime": ("Runtime s", "min"),
        "energy": ("Energy kWh", "min"),
        "co2": ("CO₂ kg", "min"),
    }
    for key, (col, direction) in specs.items():
        data = frame[["Framework", col]].dropna()
        if data.empty:
            leaders[key] = None
            continue
        idx = data[col].idxmax() if direction == "max" else data[col].idxmin()
        leaders[key] = str(frame.loc[idx, "Framework"])
    return leaders


def tradeoff_sentence(frame: pd.DataFrame) -> str:
    data = frame.dropna(subset=["Accuracy", "Energy kWh"]).copy()
    if data.empty:
        return "Measured accuracy–energy trade-off is unavailable for this run."
    acc_idx = data["Accuracy"].idxmax()
    energy_idx = data["Energy kWh"].idxmin()
    acc_row = data.loc[acc_idx]
    energy_row = data.loc[energy_idx]
    if acc_idx == energy_idx:
        return (
            f"{acc_row['Framework']} is both the accuracy leader and the lowest-energy "
            "framework in this run; no accuracy–energy compromise is needed between those two criteria."
        )
    pp = (float(acc_row["Accuracy"]) - float(energy_row["Accuracy"])) * 100.0
    ratio = float(acc_row["Energy kWh"]) / float(energy_row["Energy kWh"]) if float(energy_row["Energy kWh"]) > 0 else np.nan
    return (
        f"{acc_row['Framework']} gains {pp:+.2f} percentage points of accuracy over "
        f"{energy_row['Framework']}, while using {ratio:.2f}× its measured energy."
    )
