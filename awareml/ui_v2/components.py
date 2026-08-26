from __future__ import annotations

import html
import re
import streamlit as st


EVIDENCE_PATTERN = re.compile(r"\[(evidence\.[^\]]+)\]")


def hero(kicker, title, subtitle, pills=None):
    pill_html = ""
    for label, status in pills or []:
        cls = "good" if status == "good" else "warn" if status == "warn" else ""
        pill_html += '<span class="r9-pill {}">{}</span>'.format(cls, html.escape(str(label)))
    st.markdown(
        '<section class="r9-hero"><div class="r9-kicker">{}</div><div class="r9-title">{}</div><div class="r9-subtitle">{}</div><div class="r9-pills">{}</div></section>'.format(
            html.escape(kicker), html.escape(title), html.escape(subtitle), pill_html
        ),
        unsafe_allow_html=True,
    )



def section(title, subtitle=None):
    st.markdown('<div class="r9-section">{}</div>'.format(html.escape(title)), unsafe_allow_html=True)
    if subtitle:
        st.markdown('<div class="r9-section-sub">{}</div>'.format(html.escape(subtitle)), unsafe_allow_html=True)



def metric_card(label, value, note, tone=None):
    cls = "good" if tone == "good" else "warn" if tone == "warn" else ""
    st.markdown(
        '<div class="r9-card"><div class="r9-card-label">{}</div><div class="r9-card-value {}">{}</div><div class="r9-card-note">{}</div></div>'.format(
            html.escape(label), cls, html.escape(value), html.escape(note)
        ),
        unsafe_allow_html=True,
    )



def _friendly_evidence_label(key):
    key = str(key)
    if key == "evidence.before.recommendation.top_framework":
        return "Top framework"
    if key.startswith("evidence.before.candidates."):
        parts = key.split(".")
        if len(parts) >= 5:
            return "{} · {}".format(parts[3], parts[4].replace("_", " "))
    if key.startswith("evidence.before.recommendation."):
        return key.rsplit(".", 1)[-1].replace("_", " ").title()
    if key.startswith("evidence.before.goal_interpretation."):
        return "Goal · " + key.rsplit(".", 1)[-1].replace("_", " ")
    return key.replace("evidence.before.", "").replace("_", " ")



def humanize_rationale_text(text: str | None) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    def repl(match):
        return "({})".format(_friendly_evidence_label(match.group(1)))

    cleaned = EVIDENCE_PATTERN.sub(repl, text)
    cleaned = cleaned.replace("  ", " ")
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    return cleaned



def evidence_chips(keys, show_raw=False):
    if not keys:
        st.caption("No evidence keys attached.")
        return

    chips = []
    for key in keys:
        raw = str(key)
        label = _friendly_evidence_label(raw)
        chips.append(
            '<span class="r9-evidence" title="{}">{}</span>'.format(
                html.escape(raw),
                html.escape(label),
            )
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    if show_raw:
        with st.expander("Technical evidence IDs", expanded=False):
            for key in keys:
                st.code(str(key), language=None)



def status_panel(entries):
    rows = "".join(
        '<div class="r9-status-line"><span>{}</span><b>{}</b></div>'.format(
            html.escape(str(k)), html.escape(str(v))
        )
        for k, v in entries.items()
    )
    st.markdown('<div class="r9-panel">{}</div>'.format(rows), unsafe_allow_html=True)



def empty_state(title, text):
    st.markdown(
        '<div class="r9-panel"><div class="r9-card-label">CONTEXT REQUIRED</div><div style="color:var(--r9-text);font-size:17px;font-weight:650;margin-top:6px">{}</div><div class="r9-card-note" style="margin-top:8px">{}</div></div>'.format(
            html.escape(title), html.escape(text)
        ),
        unsafe_allow_html=True,
    )
