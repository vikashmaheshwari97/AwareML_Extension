from __future__ import annotations

import streamlit as st


DARK_TOKENS = {
    "bg": "#08111d",
    "bg2": "#0d1726",
    "sidebar": "#0b1420",
    "panel": "rgba(18, 26, 39, 0.92)",
    "panel_strong": "rgba(24, 35, 52, 0.96)",
    "border": "rgba(146, 168, 195, 0.20)",
    "text": "#f4f8fc",
    "muted": "#a8b6c8",
    "accent": "#79aefc",
    "accent_soft": "rgba(91, 149, 255, 0.16)",
    "good": "#53d39d",
    "warn": "#f4c76f",
    "danger": "#ff8b8b",
    "shadow": "rgba(0,0,0,0.25)",
}

LIGHT_TOKENS = {
    "bg": "#f3f6fb",
    "bg2": "#ebf1f8",
    "sidebar": "#e9eef6",
    "panel": "rgba(255,255,255,0.95)",
    "panel_strong": "rgba(255,255,255,1.0)",
    "border": "rgba(30, 41, 59, 0.12)",
    "text": "#162336",
    "muted": "#59697d",
    "accent": "#2d6cdf",
    "accent_soft": "rgba(45, 108, 223, 0.10)",
    "good": "#13875d",
    "warn": "#a66b00",
    "danger": "#cc4b4b",
    "shadow": "rgba(19, 30, 46, 0.08)",
}


def _css_vars(tokens: dict[str, str]) -> str:
    return "\n".join(f"  --r9-{k}:{v};" for k, v in tokens.items())


def inject_research_theme(theme_mode: str = "System") -> None:
    dark_vars = _css_vars(DARK_TOKENS)
    light_vars = _css_vars(LIGHT_TOKENS)
    if theme_mode == "Light":
        root_vars = light_vars
        system_override = ""
    elif theme_mode == "Dark":
        root_vars = dark_vars
        system_override = ""
    else:
        root_vars = dark_vars
        system_override = f"""
@media (prefers-color-scheme: light) {{
  :root {{
{light_vars}
  }}
}}
        """

    st.markdown(
        f"""
<style>
:root {{
{root_vars}
}}
{system_override}
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(circle at 82% 0%, rgba(86,116,173,0.14), transparent 28rem),
    radial-gradient(circle at 18% 12%, rgba(37,99,235,0.08), transparent 26rem),
    linear-gradient(180deg, var(--r9-bg2), var(--r9-bg));
}}
[data-testid="stSidebar"] {{ background:var(--r9-sidebar); border-right:1px solid var(--r9-border); }}
[data-testid="stHeader"] {{ background:color-mix(in srgb, var(--r9-bg) 82%, transparent); }}
.block-container {{ max-width:1650px; padding-top:1.35rem; padding-bottom:4rem; }}
.r9-brand{{display:flex;gap:12px;align-items:center;padding:4px 2px 16px}}
.r9-brand-mark{{display:flex;align-items:center;justify-content:center;width:44px;height:44px;border:1px solid var(--r9-border);border-radius:14px;background:var(--r9-panel-strong);box-shadow:0 10px 24px var(--r9-shadow);color:var(--r9-accent);font-size:21px;font-weight:700}}
.r9-brand-title{{color:var(--r9-text);font-size:22px;font-weight:760;letter-spacing:-.02em}}
.r9-brand-sub{{color:var(--r9-muted);font-size:11px;text-transform:uppercase;letter-spacing:.13em}}
.r9-side-context{{border:1px solid var(--r9-border);background:var(--r9-panel);border-radius:16px;padding:13px 14px;margin:7px 0 14px;box-shadow:0 12px 28px var(--r9-shadow)}}
.r9-muted-panel{{opacity:.92}}
.r9-eyebrow,.r9-nav-label{{color:var(--r9-muted);font-size:10px;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}
.r9-side-dataset{{margin-top:6px;color:var(--r9-text);font-size:15px;font-weight:700;overflow-wrap:anywhere}}
.r9-side-meta{{margin-top:4px;color:var(--r9-muted);font-size:11px;line-height:1.5}}
.r9-side-status{{display:grid;grid-template-columns:1fr auto;gap:7px 10px;margin:10px 3px 20px;font-size:11px;color:var(--r9-muted)}}
.r9-side-status b{{font-size:10px;letter-spacing:.08em}}
.r9-side-status .ok{{color:var(--r9-good)}} .r9-side-status .warn{{color:var(--r9-warn)}}
.r9-nav-label{{margin:4px 3px 8px}}
.r9-hero{{border:1px solid var(--r9-border);border-radius:24px;padding:28px 30px;background:linear-gradient(135deg,color-mix(in srgb, var(--r9-panel-strong) 88%, transparent),color-mix(in srgb, var(--r9-panel) 92%, transparent));margin-bottom:20px;overflow:hidden;position:relative;box-shadow:0 16px 36px var(--r9-shadow)}}
.r9-hero:after{{content:"";position:absolute;width:320px;height:320px;border-radius:50%;right:-90px;top:-140px;background:radial-gradient(circle, color-mix(in srgb, var(--r9-accent) 26%, transparent), transparent 70%);pointer-events:none}}
.r9-kicker{{color:var(--r9-accent);font-size:11px;font-weight:800;letter-spacing:.15em;text-transform:uppercase}}
.r9-title{{color:var(--r9-text);font-size:clamp(29px,4vw,50px);line-height:1.02;font-weight:790;letter-spacing:-.05em;margin:8px 0 12px}}
.r9-subtitle{{color:color-mix(in srgb, var(--r9-text) 70%, var(--r9-muted));max-width:920px;line-height:1.62;font-size:14px}}
.r9-pills{{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}}
.r9-pill{{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--r9-border);border-radius:999px;padding:6px 10px;color:var(--r9-text);font-size:11px;background:color-mix(in srgb, var(--r9-panel-strong) 92%, transparent)}}
.r9-pill.good:before,.r9-pill.warn:before{{content:"";width:7px;height:7px;border-radius:50%}}
.r9-pill.good:before{{background:var(--r9-good)}} .r9-pill.warn:before{{background:var(--r9-warn)}}
.r9-card{{border:1px solid var(--r9-border);border-radius:18px;background:var(--r9-panel);padding:17px 18px;min-height:112px;box-shadow:0 12px 28px var(--r9-shadow)}}
.r9-card-label{{color:var(--r9-muted);font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:800}}
.r9-card-value{{color:var(--r9-text);font-size:28px;font-weight:770;letter-spacing:-.035em;margin-top:5px}}
.r9-card-note{{color:var(--r9-muted);font-size:11px;line-height:1.5;margin-top:6px}} .r9-card .good{{color:var(--r9-good)}} .r9-card .warn{{color:var(--r9-warn)}}
.r9-section{{color:var(--r9-text);font-size:19px;font-weight:760;letter-spacing:-.02em;margin:28px 0 11px}}
.r9-section-sub{{color:var(--r9-muted);font-size:12px;margin-top:-7px;margin-bottom:13px}}
.r9-panel{{border:1px solid var(--r9-border);border-radius:18px;background:var(--r9-panel);padding:17px 18px;box-shadow:0 12px 28px var(--r9-shadow)}}
.r9-pipeline{{display:grid;grid-template-columns:repeat(7,minmax(80px,1fr));align-items:center;gap:8px;padding:6px 0}}
.r9-pipeline-node{{border:1px solid var(--r9-border);border-radius:14px;padding:12px 8px;min-height:62px;display:flex;justify-content:center;align-items:center;text-align:center;color:var(--r9-text);font-size:11px;line-height:1.35;background:color-mix(in srgb, var(--r9-panel-strong) 92%, transparent)}}
.r9-pipeline-arrow{{color:var(--r9-muted);text-align:center;font-weight:800}}
.r9-callout{{border-left:4px solid var(--r9-accent);background:var(--r9-accent_soft);border-radius:0 12px 12px 0;padding:12px 14px;color:var(--r9-text);font-size:12px;line-height:1.6}}
.r9-evidence{{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:11px;color:var(--r9-accent);background:color-mix(in srgb, var(--r9-accent) 10%, transparent);border:1px solid color-mix(in srgb, var(--r9-accent) 22%, transparent);border-radius:8px;padding:4px 7px;overflow-wrap:anywhere}}
.r9-status-line{{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid color-mix(in srgb, var(--r9-border) 85%, transparent);color:var(--r9-muted);font-size:12px}} .r9-status-line:last-child{{border-bottom:none}} .r9-status-line b{{color:var(--r9-text)}}
div[data-testid="stMetric"]{{border:1px solid var(--r9-border);background:var(--r9-panel);padding:14px 16px;border-radius:16px;box-shadow:0 12px 28px var(--r9-shadow)}}
div[data-testid="stMetric"] label{{color:var(--r9-muted)!important}}
div[data-testid="stMetricValue"]{{color:var(--r9-text)!important}}
div[data-testid="stDataFrame"],div[data-testid="stTable"]{{border:1px solid var(--r9-border);border-radius:14px;overflow:hidden;box-shadow:0 12px 28px var(--r9-shadow)}}
div[data-testid="stPlotlyChart"]{{border:1px solid var(--r9-border);border-radius:18px;overflow:hidden;background:var(--r9-panel);box-shadow:0 12px 28px var(--r9-shadow);padding:3px}}
div[data-testid="stPlotlyChart"] iframe, div[data-testid="stPlotlyChart"] > div{{max-width:100%!important;overflow:hidden!important}}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,textarea{{border-radius:11px!important;background:var(--r9-panel-strong)!important;color:var(--r9-text)!important;border-color:var(--r9-border)!important}}
textarea{{line-height:1.5}}
.stSlider [data-baseweb="slider"] [role="slider"]{{background:var(--r9-accent)!important}}
.stButton>button,.stDownloadButton>button{{border-radius:11px;min-height:40px;box-shadow:none}}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"]{{background:linear-gradient(135deg, var(--r9-accent), color-mix(in srgb, var(--r9-accent) 60%, #a78bfa)); color:white; border:none}}
[data-testid="stTabs"] [data-baseweb="tab-list"]{{gap:6px}}
[data-testid="stTabs"] [data-baseweb="tab"]{{border-radius:10px}}
[data-baseweb="radio"] label span{{color:var(--r9-text)!important}}
section.main * {{ color: inherit; }}

/* ---- High-contrast Streamlit widget overrides ---- */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label span,
[data-testid="stSidebar"] summary,
[data-testid="stSidebar"] summary p,
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [role="radiogroup"] label span {{
  color:var(--r9-text)!important;
  opacity:1!important;
  font-weight:650!important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding:7px 9px!important;
  border-radius:10px!important;
  margin:1px 0!important;
  transition:background .15s ease, transform .15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background:color-mix(in srgb, var(--r9-accent) 10%, transparent)!important;
  transform:translateX(2px);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background:color-mix(in srgb, var(--r9-accent) 15%, var(--r9-panel))!important;
  box-shadow:inset 3px 0 0 var(--r9-accent);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span {{
  color:var(--r9-accent)!important;
  font-weight:800!important;
}}
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
  color:var(--r9-muted)!important;
  opacity:1!important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
  border:1px solid var(--r9-border)!important;
  border-radius:12px!important;
  background:var(--r9-panel)!important;
}}
[data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] label p,
[data-testid="stAppViewContainer"] .stTextArea label p,
[data-testid="stAppViewContainer"] .stSelectbox label p,
[data-testid="stAppViewContainer"] .stNumberInput label p,
[data-testid="stAppViewContainer"] .stSlider label p,
[data-testid="stAppViewContainer"] .stToggle label p {{
  color:var(--r9-text)!important;
  opacity:1!important;
  font-weight:700!important;
}}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li {{
  color:var(--r9-text);
}}
[data-testid="stAppViewContainer"] small,
[data-testid="stAppViewContainer"] .stCaption p {{
  color:var(--r9-muted)!important;
  opacity:1!important;
}}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
  color:var(--r9-text)!important;
  -webkit-text-fill-color:var(--r9-text)!important;
  opacity:1!important;
}}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {{
  color:var(--r9-muted)!important;
  opacity:.9!important;
}}
.r9-field-label {{
  color:var(--r9-text);
  font-size:13px;
  font-weight:800;
  margin:6px 0 7px;
}}
.r9-field-help {{
  color:var(--r9-muted);
  font-size:11px;
  line-height:1.45;
  margin:-3px 0 8px;
}}
.r9-dataset-guide {{
  border:1px solid color-mix(in srgb, var(--r9-accent) 28%, var(--r9-border));
  background:linear-gradient(135deg, color-mix(in srgb, var(--r9-accent) 10%, var(--r9-panel)), var(--r9-panel));
  border-radius:16px;
  padding:15px 17px;
  margin:12px 0 16px;
  box-shadow:0 10px 24px var(--r9-shadow);
}}
.r9-dataset-guide b {{ color:var(--r9-text); }}
.r9-dataset-guide code {{ color:var(--r9-accent); background:color-mix(in srgb,var(--r9-accent) 9%,transparent); padding:2px 5px; border-radius:5px; }}

@media(max-width:950px){{.r9-pipeline{{grid-template-columns:1fr}}.r9-pipeline-arrow{{transform:rotate(90deg)}}}}
</style>
        """,
        unsafe_allow_html=True,
    )
