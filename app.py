"""
app.py — Jarvis Jira BDD Story Generator
A Streamlit app powered by LangGraph + RAG + OpenRouter to generate Jira user stories in BDD format.
"""
import os
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


def save_credentials_to_env(openrouter_key, model, jira_url, jira_email,
                             jira_token, jira_project, jira_issue_type):
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

# ── Claude-inspired Design System ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --cl-bg:          #FFFFFF;
    --cl-surface:     #FFFFFF;
    --cl-surface-2:   #F1F5F9;
    --cl-sidebar-bg:  #F8FAFC;
    --cl-sidebar-2:   #F1F5F9;
    --cl-sidebar-3:   #E2E8F0;

    --cl-orange:      #2563EB;
    --cl-orange-2:    #1D4ED8;
    --cl-orange-light:#EFF6FF;
    --cl-coral:       #3B82F6;

    --cl-text-1:      #0F172A;
    --cl-text-2:      #475569;
    --cl-text-3:      #94A3B8;
    --cl-text-inv:    #0F172A;
    --cl-text-inv-2:  rgba(15,23,42,0.7);

    --cl-border:      rgba(0,0,0,0.08);
    --cl-border-2:    rgba(0,0,0,0.05);
    --cl-border-sb:   rgba(0,0,0,0.08);

    --cl-success:     #15803D;
    --cl-success-bg:  #DCFCE7;
    --cl-warn:        #B45309;
    --cl-warn-bg:     #FEF3C7;
    --cl-error:       #B91C1C;
    --cl-error-bg:    #FEE2E2;

    --cl-shadow-sm:   0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --cl-shadow-md:   0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --cl-shadow-lg:   0 8px 24px rgba(0,0,0,0.05), 0 3px 8px rgba(0,0,0,0.03);
    --cl-shadow-glow: 0 0 0 3px rgba(37,99,235,0.15);

    --cl-r-sm: 8px;
    --cl-r-md: 12px;
    --cl-r-lg: 16px;
    --cl-r-xl: 20px;
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background-color: var(--cl-bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--cl-text-1) !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ── Sidebar — Light Panel ── */
[data-testid="stSidebar"] {
    background: var(--cl-sidebar-bg) !important;
    border-right: 1px solid var(--cl-border-sb) !important;
}
[data-testid="stSidebar"] * {
    color: var(--cl-text-inv) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: var(--cl-text-inv-2) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stTextArea > div > div > textarea,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--cl-sidebar-2) !important;
    border: 1px solid var(--cl-border-sb) !important;
    border-radius: var(--cl-r-sm) !important;
    color: var(--cl-text-inv) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
    border-color: var(--cl-orange) !important;
    box-shadow: var(--cl-shadow-glow) !important;
    outline: none !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--cl-border-sb) !important;
    margin: 0.75rem 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: var(--cl-sidebar-2) !important;
    border: 1px solid var(--cl-border-sb) !important;
    color: var(--cl-text-inv) !important;
    border-radius: var(--cl-r-sm) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--cl-sidebar-3) !important;
    border-color: var(--cl-orange) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--cl-orange) !important;
    border-color: var(--cl-orange) !important;
    color: white !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: var(--cl-orange-2) !important;
}

/* ── Main area inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stSlider > div {
    border: 1.5px solid var(--cl-border) !important;
    border-radius: var(--cl-r-md) !important;
    font-family: 'Inter', sans-serif !important;
    background: var(--cl-surface) !important;
    color: var(--cl-text-1) !important;
    box-shadow: var(--cl-shadow-sm) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--cl-orange) !important;
    box-shadow: var(--cl-shadow-glow) !important;
    outline: none !important;
}

/* ── Buttons (main area) ── */
.stButton > button {
    border-radius: var(--cl-r-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.1px !important;
    transition: all 0.15s ease !important;
    box-shadow: var(--cl-shadow-sm) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--cl-orange) 0%, var(--cl-orange-2) 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--cl-surface-2) !important;
    color: var(--cl-text-3) !important;
    box-shadow: none !important;
    transform: none !important;
}
.stButton > button[kind="secondary"] {
    background: var(--cl-surface) !important;
    border: 1.5px solid var(--cl-border) !important;
    color: var(--cl-text-2) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--cl-orange) !important;
    color: var(--cl-orange) !important;
    background: var(--cl-orange-light) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div { border: none !important; box-shadow: none !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--cl-orange) !important;
    border-color: var(--cl-orange) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1.5px solid var(--cl-border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--cl-text-3) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1.5px !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cl-orange) !important;
    border-bottom-color: var(--cl-orange) !important;
}

/* ── Expander ── */
details > summary {
    background: var(--cl-surface) !important;
    border: 1.5px solid var(--cl-border) !important;
    border-radius: var(--cl-r-md) !important;
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    color: var(--cl-text-1) !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--cl-orange), var(--cl-coral)) !important;
    border-radius: 99px !important;
}

/* ── Code ── */
code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    background: var(--cl-surface-2) !important;
    color: var(--cl-orange-2) !important;
    border-radius: 5px !important;
    padding: 1px 5px !important;
    font-size: 0.83em !important;
}
pre { background: var(--cl-surface-2) !important; border-radius: var(--cl-r-md) !important; }
pre code {
    background: transparent !important;
    color: var(--cl-text-1) !important;
    font-size: 0.85rem !important;
    line-height: 1.7 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 99px; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--cl-border-2) !important; margin: 0.5rem 0 !important; }

/* ── Alert / toast ── */
[data-testid="stAlert"] { border-radius: var(--cl-r-md) !important; }

/* ──────────────────────────────────────────── */
/*  Custom components                           */
/* ──────────────────────────────────────────── */

/* App header */
.jarvis-header {
    background: var(--cl-sidebar-bg);
    border-radius: var(--cl-r-xl);
    padding: 2rem 2.25rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--cl-shadow-md);
    border: 1px solid var(--cl-border);
}
.jarvis-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 90% 50%,
        rgba(37,99,235,0.08) 0%,
        rgba(37,99,235,0.02) 50%,
        transparent 100%);
    pointer-events: none;
}
.jarvis-logo {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, var(--cl-orange) 0%, var(--cl-coral) 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}
.jarvis-title {
    font-size: 1.6rem; font-weight: 700;
    color: var(--cl-text-inv); letter-spacing: -0.3px; margin: 0;
}
.jarvis-sub {
    font-size: 0.875rem; color: var(--cl-text-inv-2);
    margin: 4px 0 0; line-height: 1.5;
}

/* Pill tags in header */
.cl-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.2px;
    background: rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.08);
    color: rgba(0,0,0,0.6);
}

/* Metrics row */
.cl-metric {
    background: var(--cl-surface);
    border: 1.5px solid var(--cl-border);
    border-radius: var(--cl-r-lg);
    padding: 1.1rem 1rem 1rem;
    text-align: center;
    box-shadow: var(--cl-shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
}
.cl-metric:hover {
    box-shadow: var(--cl-shadow-md);
    transform: translateY(-2px);
}
.cl-metric-value {
    font-size: 1.65rem; font-weight: 700;
    color: var(--cl-orange); line-height: 1.1;
}
.cl-metric-label {
    font-size: 0.68rem; font-weight: 600;
    color: var(--cl-text-3); text-transform: uppercase;
    letter-spacing: 0.8px; margin-top: 5px;
}
.cl-metric-sub {
    font-size: 0.75rem; margin-top: 3px;
}

/* Section title */
.cl-section-title {
    font-size: 0.95rem; font-weight: 600;
    color: var(--cl-text-1); margin: 0 0 0.75rem;
    display: flex; align-items: center; gap: 8px;
}
.cl-section-title::after {
    content: '';
    flex: 1; height: 1px; background: var(--cl-border);
}

/* Feature input box */
.cl-input-card {
    background: var(--cl-surface);
    border: 1.5px solid var(--cl-border);
    border-radius: var(--cl-r-xl);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--cl-shadow-sm);
    margin-bottom: 1rem;
}

/* Thinking / loading */
.cl-thinking {
    display: flex; align-items: center; gap: 14px;
    padding: 1rem 1.25rem;
    background: var(--cl-surface);
    border: 1.5px solid rgba(37,99,235,0.25);
    border-radius: var(--cl-r-lg);
    margin: 0.75rem 0;
    box-shadow: var(--cl-shadow-sm);
}
.cl-thinking-text { font-size: 0.88rem; color: var(--cl-text-2); }
.cl-thinking-title { font-weight: 600; color: var(--cl-orange); font-size: 0.9rem; }

/* Dot pulse */
.cl-dots { display: flex; gap: 5px; align-items: center; }
.cl-dots span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--cl-orange); opacity: 0.8;
    animation: cl-pulse 1.4s ease-in-out infinite;
}
.cl-dots span:nth-child(2) { animation-delay: 0.18s; }
.cl-dots span:nth-child(3) { animation-delay: 0.36s; }
@keyframes cl-pulse {
    0%, 80%, 100% { transform: scale(0.55); opacity: 0.3; }
    40%           { transform: scale(1);    opacity: 1;   }
}

/* Story cards */
.cl-story-card {
    background: var(--cl-surface);
    border: 1.5px solid var(--cl-border);
    border-radius: var(--cl-r-xl);
    padding: 1.25rem 1.5rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--cl-shadow-sm);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.cl-story-card:hover { box-shadow: var(--cl-shadow-md); }
.cl-story-card-pushed {
    border-left: 4px solid var(--cl-success);
    background: linear-gradient(to right, rgba(21,128,61,0.04), var(--cl-surface));
}

/* Chips */
.cl-chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.3px;
    border: 1.5px solid; white-space: nowrap;
}
.cl-chip-orange { background: var(--cl-orange-light); color: var(--cl-orange-2); border-color: rgba(37,99,235,0.3); }
.cl-chip-green  { background: var(--cl-success-bg);   color: var(--cl-success);   border-color: rgba(21,128,61,0.3); }
.cl-chip-amber  { background: var(--cl-warn-bg);      color: var(--cl-warn);      border-color: rgba(180,83,9,0.25); }
.cl-chip-red    { background: var(--cl-error-bg);     color: var(--cl-error);     border-color: rgba(185,28,28,0.25); }
.cl-chip-muted  { background: var(--cl-surface-2);    color: var(--cl-text-2);    border-color: var(--cl-border); }

/* Quality review */
.cl-review-score {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 6px;
    padding: 1.5rem 1rem;
    background: var(--cl-surface-2);
    border-radius: var(--cl-r-lg);
}
.cl-score-number {
    font-size: 3rem; font-weight: 700; line-height: 1;
}

/* Jira link button */
.cl-jira-link {
    display: block; text-align: center; text-decoration: none;
    padding: 0.55rem 1rem;
    background: var(--cl-success-bg);
    border: 1.5px solid rgba(21,128,61,0.3);
    border-radius: var(--cl-r-md);
    color: var(--cl-success) !important;
    font-weight: 600; font-size: 0.875rem;
    transition: all 0.15s;
}
.cl-jira-link:hover {
    background: rgba(21,128,61,0.12);
    border-color: var(--cl-success);
    transform: translateY(-1px);
}

/* Empty state */
.cl-empty {
    text-align: center; padding: 3.5rem 2rem;
    background: var(--cl-surface);
    border: 2px dashed rgba(0,0,0,0.1);
    border-radius: var(--cl-r-xl);
    margin-top: 1rem;
}
.cl-empty-icon {
    width: 56px; height: 56px; margin: 0 auto 1rem;
    background: linear-gradient(135deg, var(--cl-orange) 0%, var(--cl-coral) 100%);
    border-radius: 16px; display: flex; align-items: center;
    justify-content: center; font-size: 1.5rem;
    box-shadow: 0 4px 12px rgba(37,99,235,0.2);
}
.cl-steps {
    display: flex; justify-content: center;
    flex-wrap: wrap; gap: 8px; margin-top: 1.5rem;
}
.cl-step {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 99px;
    background: var(--cl-surface-2); border: 1.5px solid var(--cl-border);
    font-size: 0.78rem; font-weight: 500; color: var(--cl-text-2);
}
.cl-step-active {
    background: var(--cl-orange); border-color: var(--cl-orange);
    color: white;
}

/* Sidebar branding */
.cl-sb-brand {
    padding: 1.25rem 1rem 0.5rem;
    display: flex; align-items: center; gap: 10px;
}
.cl-sb-logo {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, var(--cl-orange), var(--cl-coral));
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.cl-sb-name { font-size: 0.95rem; font-weight: 700; color: var(--cl-text-inv); }
.cl-sb-tagline { font-size: 0.68rem; color: var(--cl-text-inv-2); }

/* Sidebar section label */
.cl-sb-label {
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--cl-text-inv-2);
    padding: 0.5rem 0 0.4rem;
    border-bottom: 1px solid var(--cl-border-sb);
    margin-bottom: 0.6rem;
}

/* Sidebar status rows */
.cl-sb-status {
    font-size: 0.78rem; color: var(--cl-text-inv-2);
    display: flex; flex-direction: column; gap: 5px;
    padding: 0.5rem 0;
}
.cl-sb-status-row { display: flex; align-items: center; gap: 7px; }
.cl-sb-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* Config saved banner */
.cl-saved {
    background: var(--cl-success-bg);
    color: var(--cl-success);
    border: 1px solid rgba(21,128,61,0.3);
    border-radius: var(--cl-r-sm);
    padding: 6px 12px;
    font-size: 0.8rem; font-weight: 500;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "vectorstore":        None,
        "retriever":          None,
        "docs_loaded":        False,
        "docs_count":         0,
        "generated_stories":  [],
        "stories_markdown":   "",
        "review":             {},
        "jira_results":       {},
        "edited_stories":     {},
        "api_key":            settings.openrouter_api_key,
        "selected_model":     settings.default_model,
        "generation_error":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="cl-sb-brand">
        <div class="cl-sb-logo">🤖</div>
        <div>
            <div class="cl-sb-name">Jarvis</div>
            <div class="cl-sb-tagline">BDD · Jira · LangGraph · OpenRouter</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── LLM ──────────────────────────────────────────
    st.markdown('<div class="cl-sb-label">LLM Configuration</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=settings.openrouter_api_key,
        placeholder="sk-or-v1-...",
        key="openrouter_key_input",
    )
    st.session_state.api_key = api_key_input
    if api_key_input:
        os.environ["OPENROUTER_API_KEY"] = api_key_input

    model_names = list(AVAILABLE_MODELS.keys())
    model_ids   = list(AVAILABLE_MODELS.values())
    default_idx = model_ids.index(settings.default_model) if settings.default_model in model_ids else 0

    selected_model_name = st.selectbox("Model", options=model_names, index=default_idx, key="model_select")
    selected_model_id   = AVAILABLE_MODELS[selected_model_name]
    st.session_state.selected_model = selected_model_id

    custom_model = st.text_input("Custom model ID", placeholder="e.g. mistralai/mistral-large", key="custom_model_input")
    if custom_model.strip():
        selected_model_id = custom_model.strip()
        st.session_state.selected_model = selected_model_id

    st.divider()

    # ── Docs ─────────────────────────────────────────
    st.markdown('<div class="cl-sb-label">Training Documents</div>', unsafe_allow_html=True)

    docs_folder     = Path("docs")
    preloaded_files = list(docs_folder.glob("*")) if docs_folder.exists() else []
    preloaded_count = sum(1 for f in preloaded_files if not f.name.startswith("."))

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, DOCX, PPTX, TXT, MD · max 1 GB per file",
        key="doc_uploader",
    )

    build_btn = st.button("Build Knowledge Base", type="secondary", use_container_width=True, key="build_kb_btn")

    if build_btn:
        if not api_key_input:
            st.error("Enter your OpenRouter API key first.")
        else:
            with st.spinner("Indexing..."):
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
                        st.session_state.vectorstore  = vs
                        st.session_state.retriever    = get_retriever(vs)
                        st.session_state.docs_loaded  = True
                        st.session_state.docs_count   = len(all_docs)
                        st.success(f"Indexed {len(all_docs)} chunks")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()

    # ── Jira ─────────────────────────────────────────
    st.markdown('<div class="cl-sb-label">Jira Connection</div>', unsafe_allow_html=True)

    jira_url   = st.text_input("Base URL",   value=settings.jira_base_url  or "", placeholder="https://yourorg.atlassian.net", key="jira_url_input")
    jira_email = st.text_input("Email",      value=settings.jira_email     or "", placeholder="you@company.com",               key="jira_email_input")
    jira_token = st.text_input("API Token",  type="password", value=settings.jira_api_token or "", placeholder="Jira API token", key="jira_token_input")
    jira_project = st.text_input("Project Key", value=settings.jira_project_key or "", placeholder="PROJ", key="jira_project_input", max_chars=20)
    jira_issue_type = st.selectbox("Issue Type", ["Story", "Task", "Bug", "Epic", "Sub-task"], key="jira_issue_type")

    col_test, col_save = st.columns(2)
    with col_test:
        test_btn = st.button("Test", use_container_width=True, key="test_jira_btn")
    with col_save:
        save_btn = st.button("Save Config", type="primary", use_container_width=True, key="save_config_btn")

    if test_btn:
        if jira_url and jira_email and jira_token:
            with st.spinner("Testing..."):
                ok, msg = JiraClient(jira_url, jira_email, jira_token).test_connection()
                st.success(msg) if ok else st.error(msg)
        else:
            st.warning("Fill in all Jira fields first.")

    if save_btn:
        save_credentials_to_env(api_key_input, selected_model_id,
                                 jira_url, jira_email, jira_token,
                                 jira_project, jira_issue_type)
        st.markdown('<div class="cl-saved">✓ Saved to .env — auto-loads next session</div>', unsafe_allow_html=True)

    # Status
    st.divider()
    ok_key  = bool(api_key_input)
    ok_jira = bool(jira_url and jira_email and jira_token)
    ok_kb   = st.session_state.docs_loaded

    def dot(ok, warn=False):
        c = "#4CAF50" if ok else ("#FF9800" if warn else "#666")
        return f'<span class="cl-sb-dot" style="background:{c};"></span>'

    st.markdown(f"""
    <div class="cl-sb-status">
        <div class="cl-sb-status-row">{dot(ok_key)} OpenRouter API Key {"configured" if ok_key else "not set"}</div>
        <div class="cl-sb-status-row">{dot(ok_jira)} Jira {"connected" if ok_jira else "not configured"}</div>
        <div class="cl-sb-status-row">{dot(ok_kb, warn=True)} Knowledge Base {"ready" if ok_kb else "optional"}</div>
    </div>
    <div style="font-size:0.62rem; color:#44403C; text-align:center; padding:0.75rem 0 0.25rem;">
        Jarvis v2.0 · LangGraph · OpenRouter · FAISS
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════

# ── App Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="jarvis-header">
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:10px; position:relative;">
        <div class="jarvis-logo">🤖</div>
        <div>
            <h1 class="jarvis-title">Jarvis Story Agent</h1>
            <p class="jarvis-sub">
                Generate production-ready Jira user stories in BDD format,
                grounded in your documentation and powered by AI.
            </p>
        </div>
    </div>
    <div style="display:flex; gap:6px; flex-wrap:wrap; position:relative; margin-top:0.75rem;">
        <span class="cl-pill">LangGraph Agent</span>
        <span class="cl-pill">RAG · FAISS</span>
        <span class="cl-pill">BDD / Gherkin</span>
        <span class="cl-pill">Jira REST v3</span>
        <span class="cl-pill">10+ LLM Models</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics Row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    kb_val   = "Ready" if st.session_state.docs_loaded else "—"
    kb_color = "#2D6A4F" if st.session_state.docs_loaded else "#A8A29E"
    kb_sub   = f'<div class="cl-metric-sub" style="color:{"#2D6A4F" if st.session_state.docs_loaded else "#ED6C02"};">{"✓ Indexed" if st.session_state.docs_loaded else "Not built yet"}</div>'
    st.markdown(f'<div class="cl-metric"><div class="cl-metric-value" style="color:{kb_color}; font-size:1.15rem;">{kb_val}</div><div class="cl-metric-label">Knowledge Base</div>{kb_sub}</div>', unsafe_allow_html=True)

with c2:
    n = len(st.session_state.generated_stories)
    st.markdown(f'<div class="cl-metric"><div class="cl-metric-value">{n if n else "—"}</div><div class="cl-metric-label">Stories Generated</div></div>', unsafe_allow_html=True)

with c3:
    pushed = sum(1 for r in st.session_state.jira_results.values() if r.get("success"))
    p_color = "#2D6A4F" if pushed else "#A8A29E"
    st.markdown(f'<div class="cl-metric"><div class="cl-metric-value" style="color:{p_color};">{pushed if pushed else "—"}</div><div class="cl-metric-label">Pushed to Jira</div></div>', unsafe_allow_html=True)

with c4:
    score = st.session_state.review.get("quality_score", 0) if st.session_state.review else 0
    s_color = "#2D6A4F" if score >= 80 else "#9A5C00" if score >= 60 else "#9B1C1C" if score > 0 else "#A8A29E"
    st.markdown(f'<div class="cl-metric"><div class="cl-metric-value" style="color:{s_color};">{score if score else "—"}</div><div class="cl-metric-label">Quality Score</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

# ── Feature Input ─────────────────────────────────────────────────────────────
st.markdown('<div class="cl-section-title">Describe your feature</div>', unsafe_allow_html=True)

feature_input = st.text_area(
    "Feature description",
    height=118,
    placeholder=(
        "Example: Add a real-time notification system so users receive instant alerts "
        "when their case status changes, a new message arrives, or an SLA deadline approaches."
    ),
    key="feature_input",
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([3, 1.2, 0.8])
with col_a:
    num_stories = st.slider("Stories to generate", min_value=1, max_value=5, value=1, key="num_stories_slider")
with col_b:
    has_stories  = bool(st.session_state.generated_stories)
    generate_btn = st.button(
        "✓ Stories Generated" if has_stories else "⚡  Generate Stories",
        type="secondary" if has_stories else "primary",
        use_container_width=True,
        key="generate_btn",
        disabled=has_stories,
    )
with col_c:
    clear_btn = st.button("Clear", type="secondary", use_container_width=True, key="clear_btn")

if clear_btn:
    for key in ["generated_stories", "stories_markdown", "review", "jira_results",
                "edited_stories", "generation_error"]:
        st.session_state[key] = [] if key in ("generated_stories",) else ({} if key in ("review", "jira_results", "edited_stories") else "")
    st.rerun()

# ── Generation logic ──────────────────────────────────────────────────────────
if generate_btn:
    if not api_key_input:
        st.error("Please enter your OpenRouter API key in the sidebar.")
    elif not feature_input.strip():
        st.warning("Please describe the feature or requirement.")
    else:
        thinking_ph = st.empty()
        thinking_ph.markdown(f"""
        <div class="cl-thinking">
            <div class="cl-dots"><span></span><span></span><span></span></div>
            <div>
                <div class="cl-thinking-title">Jarvis is working…</div>
                <div class="cl-thinking-text">
                    Retrieving context &rarr; Generating BDD stories &rarr; Quality review
                    &nbsp;<span style="color:#A8A29E; font-size:0.8rem;">via {selected_model_id}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0, text="Initialising agent…")

        try:
            progress.progress(10, text="Retrieving context from docs…")
            time.sleep(0.25)

            result = agent_app.invoke({
                "user_input":      feature_input.strip(),
                "num_stories":     num_stories,
                "model":           selected_model_id,
                "retriever":       st.session_state.retriever,
                "context":         "",
                "stories_markdown":"",
                "stories":         [],
                "review":          {},
                "error":           "",
            })

            progress.progress(80, text="Running quality review…")
            time.sleep(0.25)
            progress.progress(100, text="Done!")
            time.sleep(0.35)
            progress.empty()
            thinking_ph.empty()

            if result.get("error"):
                st.session_state.generation_error = result["error"]
                st.error(f"Generation failed: {result['error']}")
            else:
                st.session_state.generated_stories = result.get("stories", [])
                st.session_state.stories_markdown  = result.get("stories_markdown", "")
                st.session_state.review            = result.get("review", {})
                st.session_state.jira_results      = {}
                st.session_state.edited_stories    = {str(s["index"]): s["markdown"] for s in result.get("stories", [])}
                st.session_state.generation_error  = ""
                st.rerun()

        except Exception as e:
            progress.empty()
            thinking_ph.empty()
            st.session_state.generation_error = str(e)
            st.error(f"Unexpected error: {e}")


# ── Quality Review Panel ──────────────────────────────────────────────────────
if st.session_state.review:
    rv        = st.session_state.review
    score     = rv.get("quality_score", 0)
    issues    = rv.get("issues", [])
    approved  = rv.get("approved", False)

    s_color = "#2D6A4F" if score >= 80 else "#9A5C00" if score >= 60 else "#9B1C1C"
    status  = "Approved" if approved else "Needs review"
    chip_cl = "cl-chip-green" if approved else "cl-chip-red"

    with st.expander(f"Quality Review — {score}/100  ·  {status}", expanded=not approved):
        col_s, col_d = st.columns([1, 3])
        with col_s:
            st.markdown(f"""
            <div class="cl-review-score">
                <div class="cl-score-number" style="color:{s_color};">{score}</div>
                <div style="font-size:0.72rem; color:#A8A29E;">out of 100</div>
                <span class="cl-chip {chip_cl}" style="margin-top:4px;">{status}</span>
            </div>""", unsafe_allow_html=True)
        with col_d:
            if issues:
                st.markdown("**Issues found:**")
                for i in issues:
                    st.markdown(f"- {i}")
            suggestions = rv.get("suggestions", [])
            if suggestions:
                st.markdown("**Suggestions:**")
                for s in suggestions:
                    st.markdown(f"- {s}")
            if not issues and not suggestions:
                st.success("All quality checks passed.")

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ── Generated Stories ─────────────────────────────────────────────────────────
if st.session_state.generated_stories:
    stories = st.session_state.generated_stories
    total   = len(stories)

    st.markdown(f'<div class="cl-section-title">Generated Stories &nbsp;<span style="font-weight:400; color:#A8A29E;">({total})</span></div>', unsafe_allow_html=True)

    # Batch push button
    all_pushed = all(st.session_state.jira_results.get(str(s["index"]), {}).get("success") for s in stories)
    if not all_pushed and jira_url and jira_email and jira_token and jira_project:
        if st.button(f"🚀  Push All {total} Stories to Jira", type="primary", key="push_all_btn", use_container_width=True):
            with st.spinner("Pushing all stories to Jira…"):
                for story in stories:
                    idx = str(story["index"])
                    if not st.session_state.jira_results.get(idx, {}).get("success"):
                        edited_md = st.session_state.edited_stories.get(idx, story["markdown"])
                        res = push_story_to_jira(
                            story={**story, "markdown": edited_md},
                            project_key=jira_project, issue_type=jira_issue_type,
                            jira_base_url=jira_url, jira_email=jira_email, jira_api_token=jira_token,
                        )
                        st.session_state.jira_results[idx] = res
            st.rerun()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Individual story cards ────────────────────────────────────────────────
    PRIORITY_CHIP = {
        "High":     "cl-chip-red",
        "Critical": "cl-chip-red",
        "Medium":   "cl-chip-amber",
        "Low":      "cl-chip-green",
        "Lowest":   "cl-chip-muted",
    }

    for story in stories:
        idx          = str(story["index"])
        jira_res     = st.session_state.jira_results.get(idx, {})
        pushed       = jira_res.get("success", False)
        priority     = story.get("priority", "Medium")
        pri_chip     = PRIORITY_CHIP.get(priority, "cl-chip-amber")
        pushed_class = " cl-story-card-pushed" if pushed else ""

        pushed_chip = (
            f'<a href="{jira_res.get("url","#")}" target="_blank" style="text-decoration:none;">'
            f'<span class="cl-chip cl-chip-green">'
            f'✓ {jira_res.get("key","")} — View in Jira</span></a>'
        ) if pushed else ""

        st.markdown(f"""
        <div class="cl-story-card{pushed_class}">
            <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:8px; flex-wrap:wrap; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:7px; flex-wrap:wrap;">
                    <span class="cl-chip cl-chip-orange">Story {story["index"]}/{total}</span>
                    <span class="cl-chip {pri_chip}">{priority}</span>
                    <span class="cl-chip cl-chip-muted">{story.get("story_points","?")} pts</span>
                </div>
                <div>{pushed_chip}</div>
            </div>
            <div style="font-size:0.975rem; font-weight:600; color:var(--cl-text-1); line-height:1.4;">
                {story.get("title","")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_prev, tab_edit = st.tabs([f"Preview  ·  Story {story['index']}", f"Edit  ·  Story {story['index']}"])

        with tab_prev:
            current_md = st.session_state.edited_stories.get(idx, story["markdown"])
            st.markdown(current_md)
            if jira_res.get("error") and not pushed:
                st.info(f"Jira note: {jira_res['error']}")

        with tab_edit:
            edited = st.text_area(
                "Edit",
                value=st.session_state.edited_stories.get(idx, story["markdown"]),
                height=440,
                key=f"edit_story_{idx}",
                label_visibility="collapsed",
            )
            if edited != st.session_state.edited_stories.get(idx, story["markdown"]):
                st.session_state.edited_stories[idx] = edited

        col_dl, col_push = st.columns(2)
        with col_dl:
            st.download_button(
                label=f"Download Story {story['index']}",
                data=st.session_state.edited_stories.get(idx, story["markdown"]),
                file_name=f"story_{story['index']}_bdd.md",
                mime="text/markdown",
                key=f"dl_{idx}",
                use_container_width=True,
            )
        with col_push:
            if pushed:
                st.markdown(
                    f'<a href="{jira_res.get("url","#")}" target="_blank" class="cl-jira-link">'
                    f'✓ {jira_res.get("key","Pushed")} — Open in Jira</a>',
                    unsafe_allow_html=True,
                )
            else:
                jira_ok = bool(jira_url and jira_email and jira_token and jira_project)
                push_btn = st.button(
                    f"Push Story {story['index']} to Jira",
                    type="primary",
                    use_container_width=True,
                    key=f"push_{idx}",
                    disabled=not jira_ok,
                    help="Configure Jira in the sidebar." if not jira_ok else "",
                )
                if push_btn:
                    with st.spinner(f"Pushing Story {story['index']}…"):
                        current_md = st.session_state.edited_stories.get(idx, story["markdown"])
                        res = push_story_to_jira(
                            story={**story, "markdown": current_md},
                            project_key=jira_project, issue_type=jira_issue_type,
                            jira_base_url=jira_url, jira_email=jira_email, jira_api_token=jira_token,
                        )
                        st.session_state.jira_results[idx] = res
                        if res["success"]:
                            st.success(f"Created {res['key']}")
                        else:
                            st.error(f"Failed: {res['error']}")
                    st.rerun()

        st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
elif not st.session_state.generation_error:
    st.markdown("""
    <div class="cl-empty">
        <div class="cl-empty-icon">🤖</div>
        <div style="font-size:1.05rem; font-weight:600; color:var(--cl-text-1); margin-bottom:6px;">
            Ready to generate your stories
        </div>
        <div style="color:var(--cl-text-2); max-width:440px; margin:0 auto; line-height:1.65; font-size:0.875rem;">
            Describe your feature below and click <strong>Generate Stories</strong>.
            Load your documentation from the sidebar for richer, context-aware stories.
        </div>
        <div class="cl-steps">
            <span class="cl-step">① Set API Key</span>
            <span class="cl-step">② Pick Model</span>
            <span class="cl-step">③ Load Docs</span>
            <span class="cl-step">④ Describe Feature</span>
            <span class="cl-step cl-step-active">⑤ Generate ⚡</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
