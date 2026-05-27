"""
app.py — Agentic Jira BDD Story Generator
A Streamlit app powered by LangGraph + RAG + OpenRouter to generate Jira user stories in BDD format.
"""
import os
import re
import time
import streamlit as st
from pathlib import Path

from config import settings, AVAILABLE_MODELS
from rag_engine import (
    load_docs_from_folder,
    load_docs_from_uploads,
    get_embeddings,
    build_vectorstore,
    get_retriever,
)
from agent import agent_app, push_story_to_jira
from jira_client import JiraClient

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def save_credentials_to_env(
    openrouter_key: str,
    model: str,
    jira_url: str,
    jira_email: str,
    jira_token: str,
    jira_project: str,
    jira_issue_type: str,
):
    lines = [
        f"OPENROUTER_API_KEY={openrouter_key}",
        f"OPENROUTER_MODEL={model}",
        f"JIRA_BASE_URL={jira_url}",
        f"JIRA_EMAIL={jira_email}",
        f"JIRA_API_TOKEN={jira_token}",
        f"JIRA_PROJECT_KEY={jira_project}",
        f"JIRA_ISSUE_TYPE={jira_issue_type}",
    ]
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jarvis — Jira Story Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Material Design CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons+Outlined');

:root {
    --md-primary: #1565C0;
    --md-primary-light: #1976D2;
    --md-primary-dark: #0D47A1;
    --md-on-primary: #ffffff;
    --md-surface: #ffffff;
    --md-background: #F5F5F5;
    --md-on-surface: #1C1B1F;
    --md-on-surface-variant: #49454F;
    --md-outline: #CAC4D0;
    --md-outline-variant: #E7E0EC;
    --md-success: #2E7D32;
    --md-warning: #ED6C02;
    --md-error: #D32F2F;
    --md-info: #0288D1;
    --md-elevation-1: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
    --md-elevation-2: 0 3px 6px rgba(0,0,0,0.10), 0 2px 4px rgba(0,0,0,0.06);
    --md-elevation-3: 0 6px 12px rgba(0,0,0,0.08), 0 3px 6px rgba(0,0,0,0.06);
    --md-radius-sm: 8px;
    --md-radius-md: 12px;
    --md-radius-lg: 16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--md-background) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--md-on-surface);
}

[data-testid="stSidebar"] {
    background: var(--md-surface) !important;
    border-right: 1px solid var(--md-outline-variant);
}

/* ── Chips / badges ── */
.md-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 8px;
    font-size: 0.75rem; font-weight: 500; letter-spacing: 0.2px;
    border: 1px solid var(--md-outline);
    background: var(--md-surface); color: var(--md-on-surface-variant);
}
.md-chip-filled { background: var(--md-primary); color: var(--md-on-primary); border-color: transparent; }
.md-chip-success { background: rgba(46,125,50,0.1); color: var(--md-success); border-color: rgba(46,125,50,0.3); }
.md-chip-warning { background: rgba(237,108,2,0.1); color: var(--md-warning); border-color: rgba(237,108,2,0.3); }
.md-chip-error { background: rgba(211,47,47,0.1); color: var(--md-error); border-color: rgba(211,47,47,0.3); }
.md-chip-info { background: rgba(2,136,209,0.1); color: var(--md-info); border-color: rgba(2,136,209,0.3); }

/* ── Cards ── */
.md-card {
    background: var(--md-surface);
    border-radius: var(--md-radius-lg);
    box-shadow: var(--md-elevation-1);
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid var(--md-outline-variant);
    transition: box-shadow 0.2s ease;
}
.md-card:hover { box-shadow: var(--md-elevation-2); }

.md-card-header {
    background: var(--md-primary);
    color: var(--md-on-primary);
    border-radius: var(--md-radius-lg);
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--md-elevation-2);
}

/* ── Metric cards ── */
.md-metric {
    background: var(--md-surface);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-radius-md);
    padding: 1.25rem 1rem;
    text-align: center;
    box-shadow: var(--md-elevation-1);
}
.md-metric-value {
    font-size: 1.75rem; font-weight: 700;
    color: var(--md-primary);
}
.md-metric-label {
    font-size: 0.7rem; font-weight: 500;
    color: var(--md-on-surface-variant);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-top: 4px;
}

/* ── Story cards ── */
.md-story-header {
    background: var(--md-surface);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--md-elevation-1);
}
.md-story-header-pushed {
    border-left: 4px solid var(--md-success);
}

/* ── Sidebar section labels ── */
.md-section-label {
    font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--md-primary);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--md-primary);
    display: inline-block;
}

/* ── Thinking indicator ── */
.md-thinking {
    display: flex; align-items: center; gap: 12px;
    padding: 1rem 1.25rem;
    background: rgba(21,101,192,0.06);
    border: 1px solid rgba(21,101,192,0.2);
    border-radius: var(--md-radius-md);
    margin: 1rem 0;
}
.md-dot-pulse { display: flex; gap: 5px; }
.md-dot-pulse span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--md-primary);
    animation: md-pulse 1.4s ease-in-out infinite;
}
.md-dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.md-dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes md-pulse {
    0%, 80%, 100% { transform: scale(0.5); opacity: 0.3; }
    40% { transform: scale(1); opacity: 1; }
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    border: 1px solid var(--md-outline) !important;
    border-radius: var(--md-radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    background: var(--md-surface) !important;
    color: var(--md-on-surface) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--md-primary) !important;
    box-shadow: 0 0 0 2px rgba(21,101,192,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--md-radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--md-primary) !important;
    border: none !important;
    color: var(--md-on-primary) !important;
    box-shadow: var(--md-elevation-1) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--md-primary-light) !important;
    box-shadow: var(--md-elevation-2) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--md-outline) !important;
    color: var(--md-primary) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(21,101,192,0.06) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--md-surface);
    border-bottom: 1px solid var(--md-outline-variant);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: var(--md-on-surface-variant);
    font-weight: 500;
    border-bottom: 2px solid transparent;
    padding: 0.75rem 1.25rem;
}
.stTabs [aria-selected="true"] {
    color: var(--md-primary) !important;
    border-bottom: 2px solid var(--md-primary) !important;
    background: transparent !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--md-surface) !important;
    border: 1px solid var(--md-outline-variant) !important;
    border-radius: var(--md-radius-sm) !important;
}

/* ── Code ── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(21,101,192,0.06) !important;
    color: var(--md-primary-dark) !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
}
pre code { background: transparent !important; }

/* ── Divider ── */
hr { border-color: var(--md-outline-variant) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--md-background); }
::-webkit-scrollbar-thumb { background: #bbb; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #999; }

/* ── Misc ── */
.config-saved-toast {
    background: var(--md-success); color: white;
    padding: 8px 16px; border-radius: var(--md-radius-sm);
    font-size: 0.85rem; font-weight: 500;
    display: inline-block; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def init_session():
    defaults = {
        "vectorstore": None,
        "retriever": None,
        "docs_loaded": False,
        "docs_count": 0,
        "generated_stories": [],
        "stories_markdown": "",
        "review": {},
        "jira_results": {},
        "edited_stories": {},
        "api_key": settings.openrouter_api_key,
        "selected_model": settings.default_model,
        "generation_error": "",
        "last_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.25rem 0 0.75rem;">
        <div style="font-size:2rem;">🤖</div>
        <div style="font-size:1rem; font-weight:700; color:var(--md-primary); margin-top:4px;">
            Jarvis Story Agent
        </div>
        <div style="font-size:0.7rem; color:var(--md-on-surface-variant); margin-top:2px;">
            BDD &middot; Jira &middot; LangGraph &middot; OpenRouter
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── OpenRouter Configuration ───────────────────────────────────────────
    st.markdown('<div class="md-section-label">LLM Configuration</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=settings.openrouter_api_key,
        placeholder="sk-or-v1-...",
        help="Get your key at https://openrouter.ai/keys",
        key="openrouter_key_input",
    )
    st.session_state.api_key = api_key_input
    if api_key_input:
        os.environ["OPENROUTER_API_KEY"] = api_key_input

    model_names = list(AVAILABLE_MODELS.keys())
    model_ids = list(AVAILABLE_MODELS.values())
    default_model_id = settings.default_model
    default_idx = model_ids.index(default_model_id) if default_model_id in model_ids else 0

    selected_model_name = st.selectbox(
        "Model",
        options=model_names,
        index=default_idx,
        key="model_select",
    )
    selected_model_id = AVAILABLE_MODELS[selected_model_name]
    st.session_state.selected_model = selected_model_id

    st.markdown(
        f'<span class="md-chip md-chip-info" style="margin-top:2px;">{selected_model_id}</span>',
        unsafe_allow_html=True,
    )

    custom_model = st.text_input(
        "Custom model ID (optional)",
        value="",
        placeholder="e.g. mistralai/mistral-large",
        key="custom_model_input",
    )
    if custom_model.strip():
        selected_model_id = custom_model.strip()
        st.session_state.selected_model = selected_model_id
        st.markdown(
            f'<span class="md-chip md-chip-filled">{selected_model_id}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Document Management ────────────────────────────────────────────────
    st.markdown('<div class="md-section-label">Training Documents</div>', unsafe_allow_html=True)

    docs_folder = Path("docs")
    preloaded_files = list(docs_folder.glob("*")) if docs_folder.exists() else []
    preloaded_count = sum(1 for f in preloaded_files if not f.name.startswith("."))

    if preloaded_count > 0:
        st.markdown(
            f'<span class="md-chip md-chip-success">{preloaded_count} pre-loaded doc(s)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="md-chip md-chip-warning">No pre-loaded docs</span>',
            unsafe_allow_html=True,
        )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, DOCX, PPTX, TXT, MD — up to 1 GB per file",
        key="doc_uploader",
    )

    build_btn = st.button(
        "Build Knowledge Base",
        type="secondary",
        use_container_width=True,
        key="build_kb_btn",
    )

    if build_btn:
        if not api_key_input:
            st.error("Enter your OpenRouter API key first.")
        else:
            with st.spinner("Indexing documents..."):
                try:
                    all_docs = []
                    if preloaded_count > 0:
                        all_docs.extend(load_docs_from_folder(str(docs_folder)))
                    if uploaded_files:
                        for f in uploaded_files:
                            f.seek(0)
                        all_docs.extend(load_docs_from_uploads(uploaded_files))
                    if not all_docs:
                        st.warning("No documents found.")
                    else:
                        embeddings = get_embeddings(api_key=api_key_input)
                        vs = build_vectorstore(all_docs, embeddings)
                        st.session_state.vectorstore = vs
                        st.session_state.retriever = get_retriever(vs)
                        st.session_state.docs_loaded = True
                        st.session_state.docs_count = len(all_docs)
                        st.success(f"Indexed {len(all_docs)} chunks")
                except Exception as e:
                    st.error(f"Failed: {e}")

    if st.session_state.docs_loaded:
        st.markdown(
            f'<span class="md-chip md-chip-success" style="margin-top:6px;">KB Ready — {st.session_state.docs_count} chunks</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Jira Configuration ─────────────────────────────────────────────────
    st.markdown('<div class="md-section-label">Jira Connection</div>', unsafe_allow_html=True)

    jira_url = st.text_input(
        "Base URL",
        value=settings.jira_base_url or "",
        placeholder="https://yourorg.atlassian.net",
        key="jira_url_input",
    )
    jira_email = st.text_input(
        "Email",
        value=settings.jira_email or "",
        placeholder="you@company.com",
        key="jira_email_input",
    )
    jira_token = st.text_input(
        "API Token",
        type="password",
        value=settings.jira_api_token or "",
        placeholder="Your Jira API token",
        key="jira_token_input",
    )
    jira_project = st.text_input(
        "Project Key",
        value=settings.jira_project_key or "",
        placeholder="PROJ",
        key="jira_project_input",
        max_chars=20,
    )
    jira_issue_type = st.selectbox(
        "Issue Type",
        options=["Story", "Task", "Bug", "Epic", "Sub-task"],
        index=0,
        key="jira_issue_type",
    )

    col_test, col_save = st.columns(2)
    with col_test:
        test_jira_btn = st.button("Test", use_container_width=True, key="test_jira_btn")
    with col_save:
        save_config_btn = st.button("Save Config", use_container_width=True, key="save_config_btn")

    if test_jira_btn:
        if jira_url and jira_email and jira_token:
            with st.spinner("Testing..."):
                client = JiraClient(jira_url, jira_email, jira_token)
                ok, msg = client.test_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
        else:
            st.warning("Fill in all Jira fields first.")

    if save_config_btn:
        save_credentials_to_env(
            openrouter_key=api_key_input,
            model=selected_model_id,
            jira_url=jira_url,
            jira_email=jira_email,
            jira_token=jira_token,
            jira_project=jira_project,
            jira_issue_type=jira_issue_type,
        )
        st.markdown(
            '<div class="config-saved-toast">Configuration saved to .env</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Config status
    has_key = bool(api_key_input)
    has_jira = bool(jira_url and jira_email and jira_token)
    st.markdown(f"""
    <div style="font-size:0.75rem; color:var(--md-on-surface-variant); padding:0.5rem 0;">
        <div style="margin-bottom:4px;">
            {"<span style='color:var(--md-success);'>&#10003;</span>" if has_key else "<span style='color:var(--md-error);'>&#10007;</span>"}
            OpenRouter API Key
        </div>
        <div style="margin-bottom:4px;">
            {"<span style='color:var(--md-success);'>&#10003;</span>" if has_jira else "<span style='color:var(--md-error);'>&#10007;</span>"}
            Jira Connection
        </div>
        <div>
            {"<span style='color:var(--md-success);'>&#10003;</span>" if st.session_state.docs_loaded else "<span style='color:var(--md-warning);'>&#8226;</span>"}
            Knowledge Base {"ready" if st.session_state.docs_loaded else "(optional)"}
        </div>
    </div>
    <div style="font-size:0.65rem; color:#999; text-align:center; padding:0.75rem 0 0.5rem;">
        Jarvis v2.0 &middot; LangGraph &middot; OpenRouter &middot; FAISS
    </div>
    """, unsafe_allow_html=True)


# ── Main Content Area ─────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="md-card-header">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="font-size:1.75rem;">🤖</span>
        <h1 style="margin:0; font-size:1.75rem; font-weight:700; color:white;">
            Jarvis Story Agent
        </h1>
    </div>
    <p style="margin:0; font-size:0.9rem; color:rgba(255,255,255,0.85); line-height:1.5;">
        Generate production-ready Jira user stories in BDD format &mdash; powered by AI and grounded in your documentation.
    </p>
    <div style="display:flex; gap:8px; margin-top:1rem; flex-wrap:wrap;">
        <span class="md-chip" style="background:rgba(255,255,255,0.15); color:white; border-color:rgba(255,255,255,0.3);">LangGraph Agent</span>
        <span class="md-chip" style="background:rgba(255,255,255,0.15); color:white; border-color:rgba(255,255,255,0.3);">RAG Context</span>
        <span class="md-chip" style="background:rgba(255,255,255,0.15); color:white; border-color:rgba(255,255,255,0.3);">BDD / Gherkin</span>
        <span class="md-chip" style="background:rgba(255,255,255,0.15); color:white; border-color:rgba(255,255,255,0.3);">Jira Push</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    kb_icon = "check_circle" if st.session_state.docs_loaded else "pending"
    kb_color = "var(--md-success)" if st.session_state.docs_loaded else "var(--md-warning)"
    kb_text = "Ready" if st.session_state.docs_loaded else "Not built"
    st.markdown(f"""
    <div class="md-metric">
        <div class="md-metric-value" style="color:{kb_color}; font-size:1.25rem;">{kb_text}</div>
        <div class="md-metric-label">Knowledge Base</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="md-metric">
        <div class="md-metric-value">{len(st.session_state.generated_stories)}</div>
        <div class="md-metric-label">Stories Generated</div>
    </div>""", unsafe_allow_html=True)
with col3:
    pushed = sum(1 for r in st.session_state.jira_results.values() if r.get("success"))
    st.markdown(f"""
    <div class="md-metric">
        <div class="md-metric-value" style="color:var(--md-success);">{pushed}</div>
        <div class="md-metric-label">Pushed to Jira</div>
    </div>""", unsafe_allow_html=True)
with col4:
    score = st.session_state.review.get("quality_score", 0) if st.session_state.review else 0
    score_color = "var(--md-success)" if score >= 80 else "var(--md-warning)" if score >= 60 else "var(--md-error)" if score > 0 else "var(--md-on-surface-variant)"
    st.markdown(f"""
    <div class="md-metric">
        <div class="md-metric-value" style="color:{score_color};">{score if score else "—"}</div>
        <div class="md-metric-label">Quality Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ── Story Generation Form ────────────────────────────────────────────────────
st.markdown("#### Describe your feature")

feature_input = st.text_area(
    "Feature description",
    height=120,
    placeholder=(
        "Example: Add a real-time notification system so users receive instant alerts "
        "when their case status changes, a new message arrives, or an SLA deadline approaches."
    ),
    key="feature_input",
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    num_stories = st.slider(
        "Number of stories",
        min_value=1, max_value=5, value=1,
        key="num_stories_slider",
    )
with col_b:
    has_stories = bool(st.session_state.generated_stories)
    generate_btn = st.button(
        "Stories Generated" if has_stories else "Generate Stories",
        type="secondary" if has_stories else "primary",
        use_container_width=True,
        key="generate_btn",
        disabled=has_stories,
    )
with col_c:
    clear_btn = st.button(
        "Clear",
        type="secondary",
        use_container_width=True,
        key="clear_btn",
    )

if clear_btn:
    st.session_state.generated_stories = []
    st.session_state.stories_markdown = ""
    st.session_state.review = {}
    st.session_state.jira_results = {}
    st.session_state.edited_stories = {}
    st.session_state.generation_error = ""
    st.rerun()

# ── Generation ────────────────────────────────────────────────────────────────
if generate_btn:
    if not api_key_input:
        st.error("Please enter your OpenRouter API key in the sidebar.")
    elif not feature_input.strip():
        st.warning("Please describe the feature or requirement.")
    else:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(f"""
        <div class="md-thinking">
            <div class="md-dot-pulse"><span></span><span></span><span></span></div>
            <div>
                <div style="font-weight:600; color:var(--md-primary);">Agent is thinking&hellip;</div>
                <div style="font-size:0.8rem; color:var(--md-on-surface-variant); margin-top:2px;">
                    Retrieve context &rarr; Generate BDD stories &rarr; Quality review
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0, text="Initialising agent...")

        try:
            progress.progress(10, text="Retrieving context...")
            time.sleep(0.3)

            initial_state = {
                "user_input": feature_input.strip(),
                "num_stories": num_stories,
                "model": selected_model_id,
                "retriever": st.session_state.retriever,
                "context": "",
                "stories_markdown": "",
                "stories": [],
                "review": {},
                "error": "",
            }

            progress.progress(30, text=f"Generating stories via {selected_model_id}...")
            result = agent_app.invoke(initial_state)

            progress.progress(80, text="Running quality review...")
            time.sleep(0.3)
            progress.progress(100, text="Done")
            time.sleep(0.4)
            progress.empty()
            thinking_placeholder.empty()

            if result.get("error"):
                st.session_state.generation_error = result["error"]
                st.error(f"Generation failed: {result['error']}")
            else:
                st.session_state.generated_stories = result.get("stories", [])
                st.session_state.stories_markdown = result.get("stories_markdown", "")
                st.session_state.review = result.get("review", {})
                st.session_state.jira_results = {}
                st.session_state.edited_stories = {
                    str(s["index"]): s["markdown"]
                    for s in result.get("stories", [])
                }
                st.session_state.generation_error = ""
                st.rerun()

        except Exception as e:
            progress.empty()
            thinking_placeholder.empty()
            st.session_state.generation_error = str(e)
            st.error(f"Unexpected error: {e}")


# ── Quality Review ────────────────────────────────────────────────────────────
if st.session_state.review:
    review = st.session_state.review
    score = review.get("quality_score", 0)
    issues = review.get("issues", [])
    suggestions = review.get("suggestions", [])
    approved = review.get("approved", False)

    score_color = "var(--md-success)" if score >= 80 else "var(--md-warning)" if score >= 60 else "var(--md-error)"
    status_text = "Approved" if approved else "Needs Review"
    chip_class = "md-chip-success" if approved else "md-chip-error"

    with st.expander(f"Quality Review — {score}/100 | {status_text}", expanded=not approved):
        col_r1, col_r2 = st.columns([1, 3])
        with col_r1:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem;">
                <div style="font-size:2.5rem; font-weight:700; color:{score_color};">{score}</div>
                <div style="font-size:0.75rem; color:var(--md-on-surface-variant);">out of 100</div>
                <div style="margin-top:0.5rem;"><span class="md-chip {chip_class}">{status_text}</span></div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            if issues:
                st.markdown("**Issues:**")
                for issue in issues:
                    st.markdown(f"- {issue}")
            if suggestions:
                st.markdown("**Suggestions:**")
                for s in suggestions:
                    st.markdown(f"- {s}")
            if not issues and not suggestions:
                st.markdown("All checks passed. Stories meet BDD quality standards.")

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ── Generated Stories ─────────────────────────────────────────────────────────
if st.session_state.generated_stories:
    stories = st.session_state.generated_stories
    total = len(stories)

    st.markdown(f"#### Generated Stories ({total})")

    # Batch push
    all_pushed = all(
        st.session_state.jira_results.get(str(s["index"]), {}).get("success")
        for s in stories
    )
    if not all_pushed and jira_url and jira_email and jira_token and jira_project:
        if st.button(
            f"Push All {total} Stories to Jira",
            type="primary",
            key="push_all_btn",
            use_container_width=True,
        ):
            with st.spinner("Pushing all stories..."):
                for story in stories:
                    idx = str(story["index"])
                    if not st.session_state.jira_results.get(idx, {}).get("success"):
                        edited_md = st.session_state.edited_stories.get(idx, story["markdown"])
                        result = push_story_to_jira(
                            story={**story, "markdown": edited_md},
                            project_key=jira_project,
                            issue_type=jira_issue_type,
                            jira_base_url=jira_url,
                            jira_email=jira_email,
                            jira_api_token=jira_token,
                        )
                        st.session_state.jira_results[idx] = result
            st.rerun()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Individual story cards
    for story in stories:
        idx = str(story["index"])
        jira_result = st.session_state.jira_results.get(idx, {})
        already_pushed = jira_result.get("success", False)

        priority = story.get("priority", "Medium")
        priority_colors = {"High": "var(--md-error)", "Medium": "var(--md-warning)", "Low": "var(--md-success)", "Critical": "var(--md-error)", "Lowest": "#9E9E9E"}
        p_color = priority_colors.get(priority, "var(--md-warning)")

        pushed_class = " md-story-header-pushed" if already_pushed else ""
        pushed_chip = (
            f'<span class="md-chip md-chip-success">'
            f'<a href="{jira_result.get("url","#")}" target="_blank" style="color:var(--md-success); text-decoration:none;">'
            f'{jira_result.get("key","")} — View in Jira</a></span>'
            if already_pushed else ""
        )

        st.markdown(f"""
        <div class="md-story-header{pushed_class}">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="md-chip md-chip-filled">Story {story['index']}/{total}</span>
                    <span class="md-chip" style="color:{p_color}; border-color:{p_color};">{priority}</span>
                    <span class="md-chip">{story.get('story_points', '?')} pts</span>
                </div>
                <div>{pushed_chip}</div>
            </div>
            <div style="font-size:1rem; font-weight:600; color:var(--md-on-surface);">
                {story.get('title','')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_preview, tab_edit = st.tabs([f"Preview — Story {story['index']}", f"Edit — Story {story['index']}"])

        with tab_preview:
            current_md = st.session_state.edited_stories.get(idx, story["markdown"])
            st.markdown(current_md)
            if jira_result.get("error") and not already_pushed:
                st.warning(f"Jira note: {jira_result['error']}")

        with tab_edit:
            edited = st.text_area(
                f"Edit Story {story['index']}",
                value=st.session_state.edited_stories.get(idx, story["markdown"]),
                height=450,
                key=f"edit_story_{idx}",
                label_visibility="collapsed",
            )
            if edited != st.session_state.edited_stories.get(idx, story["markdown"]):
                st.session_state.edited_stories[idx] = edited

        col_dl, col_approve = st.columns(2)
        with col_dl:
            current_md = st.session_state.edited_stories.get(idx, story["markdown"])
            st.download_button(
                label=f"Download Story {story['index']}",
                data=current_md,
                file_name=f"story_{story['index']}_bdd.md",
                mime="text/markdown",
                key=f"dl_{idx}",
                use_container_width=True,
            )

        with col_approve:
            if already_pushed:
                st.markdown(f"""
                <a href="{jira_result.get('url','#')}" target="_blank" style="
                    display:block; text-align:center;
                    padding:0.6rem 1rem;
                    background:rgba(46,125,50,0.08);
                    border:1px solid rgba(46,125,50,0.3);
                    border-radius:var(--md-radius-sm);
                    color:var(--md-success);
                    font-weight:600; text-decoration:none;
                    font-size:0.875rem;
                ">{jira_result.get('key','Pushed')} — View in Jira</a>
                """, unsafe_allow_html=True)
            else:
                jira_configured = bool(jira_url and jira_email and jira_token and jira_project)
                push_btn = st.button(
                    f"Push Story {story['index']} to Jira",
                    type="primary",
                    use_container_width=True,
                    key=f"push_{idx}",
                    disabled=not jira_configured,
                    help="Configure Jira in the sidebar to enable." if not jira_configured else "",
                )
                if push_btn:
                    with st.spinner(f"Pushing Story {story['index']}..."):
                        current_md = st.session_state.edited_stories.get(idx, story["markdown"])
                        result = push_story_to_jira(
                            story={**story, "markdown": current_md},
                            project_key=jira_project,
                            issue_type=jira_issue_type,
                            jira_base_url=jira_url,
                            jira_email=jira_email,
                            jira_api_token=jira_token,
                        )
                        st.session_state.jira_results[idx] = result
                        if result["success"]:
                            st.success(f"Created {result['key']}")
                        else:
                            st.error(f"Failed: {result['error']}")
                    st.rerun()

        st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
elif not st.session_state.generation_error:
    st.markdown("""
    <div style="
        text-align:center; padding:3rem 2rem;
        background:var(--md-surface);
        border:1px dashed var(--md-outline);
        border-radius:var(--md-radius-lg);
        margin-top:1rem;
    ">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">🤖</div>
        <div style="font-size:1.1rem; font-weight:600; color:var(--md-on-surface); margin-bottom:0.5rem;">
            Ready to generate
        </div>
        <div style="color:var(--md-on-surface-variant); max-width:480px; margin:0 auto; line-height:1.6; font-size:0.9rem;">
            Describe your feature above and click <strong>Generate Stories</strong>.
            Connect your Jira and load documentation from the sidebar for best results.
        </div>
        <div style="margin-top:1.5rem; display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
            <span class="md-chip">1. Set API Key</span>
            <span class="md-chip">2. Pick Model</span>
            <span class="md-chip">3. Load Docs</span>
            <span class="md-chip">4. Describe Feature</span>
            <span class="md-chip md-chip-filled">5. Generate</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
