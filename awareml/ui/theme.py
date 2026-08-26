import html
import streamlit as st


def inject_theme():
    st.markdown(
        """
<style>
:root {
  --ink:#10213f;
  --ink-soft:#30425f;
  --muted:#6d7890;
  --line:#e5eaf3;
  --line-strong:#d6deeb;
  --panel:#ffffff;
  --soft:#f7f9fc;
  --accent:#5b5bd6;
  --accent-soft:#eef0ff;
  --accent2:#0b9c8d;
  --accent2-soft:#eafaf7;
  --danger:#c74343;
  --warning:#a96c00;
}

html, body, [class*="css"] { font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 92% 0%, rgba(15,159,143,.065), transparent 27rem),
    radial-gradient(circle at 35% 0%, rgba(91,91,214,.055), transparent 30rem),
    linear-gradient(180deg,#fcfdff 0%,#f7f9fd 100%);
  color:var(--ink);
}
[data-testid="stHeader"] { background:rgba(255,255,255,.72); backdrop-filter:blur(12px); }
.block-container { max-width: 1540px; padding-top:1.7rem; padding-bottom:4rem; }

/* Sidebar */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#111a2e 0%,#16213a 58%,#101827 100%);
  border-right:1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color:#dce5f7 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding:.35rem .45rem;
  border-radius:10px;
}
.brand-mark {
  width:40px;height:40px;display:flex;align-items:center;justify-content:center;
  border-radius:13px;background:linear-gradient(135deg,#7272ff,#21b8a8);
  box-shadow:0 10px 28px rgba(91,91,214,.32);font-size:23px;color:white;margin-top:4px;
}
.brand-title {font-size:27px;font-weight:820;letter-spacing:-.04em;color:#fff;margin:8px 0 1px;}
.brand-sub {font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#a9b7d0;margin-bottom:8px;}


.sidebar-dataset {margin-top:14px;padding:12px 13px;border-radius:14px;background:rgba(255,255,255,.065);border:1px solid rgba(255,255,255,.08);}
.sidebar-dataset.empty {background:rgba(255,255,255,.035);}
.sidebar-kicker {font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:#92a6c8;font-weight:850;margin-bottom:5px;}
.sidebar-dataset-name {font-size:13px;font-weight:800;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.sidebar-dataset-meta {font-size:10px;color:#b9c7dc;line-height:1.45;margin-top:4px;}
.context-strip {padding:11px 14px;border:1px solid var(--line);border-radius:13px;background:linear-gradient(90deg,#f7f9ff,#f5fbfa);color:var(--ink-soft);font-size:12px;margin:-6px 0 18px;box-shadow:0 4px 14px rgba(22,35,61,.025);}
.context-strip b {color:var(--ink);}
.context-strip span {color:#55657d;}
.evidence-card {padding:15px 17px;border-radius:15px;background:linear-gradient(135deg,#f7f8ff,#fbfffe);border:1px solid var(--line);color:var(--ink-soft);line-height:1.6;margin:10px 0 14px;}
.evidence-card b {color:var(--ink);}
.evidence-card span {font-size:12px;color:var(--muted);}

/* Hero */
.hero {
  position:relative;overflow:hidden;padding:30px 34px;border:1px solid var(--line);
  border-radius:24px;background:linear-gradient(122deg,rgba(255,255,255,.98),rgba(245,246,255,.96) 58%,rgba(237,252,249,.96));
  box-shadow:0 12px 35px rgba(25,40,75,.045);margin-bottom:22px;
}
.hero:after {content:"";position:absolute;right:-75px;top:-95px;width:240px;height:240px;border-radius:999px;border:34px solid rgba(91,91,214,.045);}
.hero h1 {font-size:clamp(34px,3vw,46px);line-height:1.05;letter-spacing:-.048em;margin:0 0 10px;color:var(--ink);font-weight:830;}
.hero p {font-size:16px;line-height:1.65;color:#5b677d;max-width:920px;margin:0;}
.kicker {text-transform:uppercase;letter-spacing:.13em;font-size:10px;font-weight:850;color:var(--accent);margin-bottom:10px;}

/* Cards / metrics */
.card {padding:20px 21px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.93);box-shadow:0 6px 20px rgba(23,34,59,.035);height:100%;}
.card .label {color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.075em;}
.card .value {font-size:27px;font-weight:820;color:var(--ink);margin:5px 0;letter-spacing:-.035em;}
.card .note {font-size:12px;color:var(--muted);line-height:1.5;}
.section-title {font-size:24px;font-weight:820;color:var(--ink);letter-spacing:-.03em;margin:24px 0 12px;}
.badge {display:inline-block;padding:5px 9px;border-radius:999px;background:var(--accent-soft);color:#4f46d8;font-size:11px;font-weight:800;margin-right:5px;}
.research-warning {padding:14px 16px;border-radius:14px;background:#fff9e9;border:1px solid #f0d78d;color:#73500d;}

[data-testid="stMetric"] {
  background:rgba(255,255,255,.95);border:1px solid var(--line);padding:13px 16px;border-radius:16px;
  box-shadow:0 4px 13px rgba(22,35,61,.025);
}
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--ink); letter-spacing:-.035em; }

/* Inputs */
.stButton>button {border-radius:11px;font-weight:720;border:1px solid var(--line-strong);min-height:2.55rem;}
.stButton>button[kind="primary"] {background:linear-gradient(135deg,#5b5bd6,#6868e8);border-color:#5b5bd6;box-shadow:0 6px 15px rgba(91,91,214,.18);}
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stFileUploaderDropzone"] { border-radius:11px !important; }
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:14px;overflow:hidden;background:white;}
[data-testid="stPlotlyChart"] {background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:18px;padding:8px;box-shadow:0 8px 24px rgba(22,35,61,.035);}

/* Tabs */
button[data-baseweb="tab"] {font-weight:700;padding-left:.85rem;padding-right:.85rem;}
[data-baseweb="tab-list"] {gap:.15rem;border-bottom:1px solid var(--line);}

/* Alerts */
[data-testid="stAlert"] {border-radius:13px;}
hr {border-color:var(--line) !important;}

/* Small screens */
@media (max-width: 900px) {
  .block-container {padding-left:1rem;padding-right:1rem;}
  .hero {padding:23px 22px;border-radius:19px;}
  .hero h1 {font-size:34px;}
}
</style>
""",
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, subtitle: str):
    kicker = html.escape(str(kicker))
    title = html.escape(str(title))
    subtitle = html.escape(str(subtitle))
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
