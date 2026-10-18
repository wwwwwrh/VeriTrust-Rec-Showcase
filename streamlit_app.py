"""VeriTrust-Rec Summer Edition Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from recommender.pipeline import run_user_pipeline

from app.components import (
    render_evidence,
    render_intent,
    render_recommendations,
    render_status,
    render_trace,
)
from app.i18n import text
from app.runtime import load_runtime_assets
from app.styles import APP_CSS


st.set_page_config(page_title="VeriTrust-Rec", page_icon="🧭", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)


def _initialize_state() -> None:
    defaults = {
        "last_query": "",
        "query_input": "",
        "last_result": None,
        "ui_language": "zh",
        "debug_mode": False,
        "top_k": 5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_result() -> None:
    st.session_state["last_result"] = None


def _clear_query() -> None:
    st.session_state["query_input"] = ""
    st.session_state["last_query"] = ""
    st.session_state["last_result"] = None


_initialize_state()

with st.sidebar:
    language_label = st.selectbox(
        "Language / 语言",
        options=("zh", "en"),
        format_func=lambda value: "中文" if value == "zh" else "English",
        key="ui_language",
        on_change=_clear_result,
    )
    st.markdown("**Runtime**: Ollama  ")
    st.markdown("**Model**: `qwen3:8b`  ")
    st.markdown("**Mode**: Local")
    top_k = st.slider("Top-K Candidates", 3, 10, key="top_k")
    st.markdown("---")
    st.markdown("Evidence Retrieval ✅")
    st.markdown("Claim Verification ✅")
    st.markdown("Confidence Estimation ✅")
    st.markdown("Selective Abstention ✅")
    debug = st.checkbox(text(language_label, "debug"), key="debug_mode")

st.title(text(language_label, "title"))
st.markdown(
    f"<div class='vtr-subtitle'>{text(language_label, 'subtitle')}</div>",
    unsafe_allow_html=True,
)

query_tab, recommendations_tab, evidence_tab, trace_tab = st.tabs([
    text(language_label, "query_tab"),
    text(language_label, "recommendations_tab"),
    text(language_label, "evidence_tab"),
    text(language_label, "trace_tab"),
])

with query_tab:
    with st.form("query_form"):
        query = st.text_area(
            text(language_label, "query_label"),
            placeholder=text(language_label, "query_placeholder"),
            height=140,
            key="query_input",
        )
        submit = st.form_submit_button(text(language_label, "submit"), type="primary")
        st.form_submit_button(text(language_label, "clear"), on_click=_clear_query)
    if submit:
        st.session_state["last_query"] = query
        try:
            with st.spinner(text(language_label, "working")):
                assets = load_runtime_assets()
                st.session_state["last_result"] = run_user_pipeline(
                    query,
                    output_language=language_label,
                    top_k=top_k,
                    query_client=assets.query_client,
                    final_client=assets.final_client,
                    config=assets.config,
                    prompts=assets.prompts,
                    resources=assets.resources,
                    provenance_records=assets.provenance_records,
                    evidence_corpus=assets.evidence_corpus,
                )
        except Exception as exc:
            st.error(f"Application startup failed: {type(exc).__name__}: {exc}")
    current = st.session_state["last_result"]
    if current is None:
        st.info(text(language_label, "no_result"))
    else:
        render_status(current, language_label)
        render_intent(current.intent, language_label)

with recommendations_tab:
    current = st.session_state["last_result"]
    if current is None:
        st.info(text(language_label, "no_result"))
    else:
        render_recommendations(current, language_label)

with evidence_tab:
    current = st.session_state["last_result"]
    if current is None:
        st.info(text(language_label, "no_result"))
    else:
        excerpt_chars = 500
        try:
            excerpt_chars = load_runtime_assets().config.ui.evidence_excerpt_chars
        except Exception:
            pass
        render_evidence(current, excerpt_chars)

with trace_tab:
    current = st.session_state["last_result"]
    if current is None:
        st.info(text(language_label, "no_result"))
    else:
        render_trace(current, debug)
