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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Superman Jira Story Generator",
    page_icon="🦸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Mankind theme ── */
:root {
    --bg-primary: #f8f9fa;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #f0f7fb;
    --accent-purple: #1385C7;
    --accent-blue: #005C9E;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-subtle: rgba(19, 133, 199, 0.2);
    --gradient-hero: linear-gradient(135deg, #1385C7 0%, #005C9E 100%);
    --gradient-card: linear-gradient(145deg, #ffffff, #f8f9fa);
    --shadow-glow: 0 4px 20px rgba(19, 133, 199, 0.15);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle);
}

/* ── Hero header ── */
.hero-header {
    background: var(--gradient-hero);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow);
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(20px, -10px); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-amber { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.badge-blue { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }

/* ── Story cards ── */
.story-card {
    background: var(--gradient-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.story-card:hover {
    border-color: var(--accent-purple);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.story-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}
.story-number {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent-purple);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Sidebar sections ── */
.sidebar-section {
    background: rgba(124,58,237,0.08);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--accent-purple);
    margin-bottom: 0.75rem;
}

/* ── Input styling ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--gradient-hero) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(124,58,237,0.1) !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--accent-purple) !important;
}

/* ── Progress & metrics ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

/* ── Agent thinking indicator ── */
.thinking-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 1.25rem;
    background: rgba(124,58,237,0.08);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    margin: 1rem 0;
}
.dot-pulse {
    display: flex;
    gap: 6px;
}
.dot-pulse span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-purple);
    animation: pulse 1.4s ease-in-out infinite;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

/* ── Divider ── */
hr { border-color: var(--border-subtle) !important; }

/* ── Code blocks ── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(124,58,237,0.1) !important;
    color: #a78bfa !important;
    border-radius: 4px !important;
}
pre code { background: transparent !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-secondary);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-purple) !important;
    color: white !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-purple); border-radius: 3px; }
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
        "jira_results": {},    # story_index -> jira result dict
        "edited_stories": {},  # story_index -> edited markdown
        "api_key": "",
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
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🦸</div>
        <div style="font-size:1.1rem; font-weight:700; color:#a78bfa;">Superman Story Agent</div>
        <div style="font-size:0.75rem; color:#64748b;">BDD • Jira • LangGraph • OpenRouter</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── OpenRouter Configuration ───────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">🤖 OpenRouter LLM</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=settings.openrouter_api_key,
        placeholder="sk-or-v1-...",
        help="Get your key at: https://openrouter.ai/keys — gives you access to GPT-4o, Claude, Gemini, Llama, etc.",
        key="openrouter_key_input",
    )
    st.session_state.api_key = api_key_input
    if api_key_input:
        os.environ["OPENROUTER_API_KEY"] = api_key_input

    # Model selector
    model_names = list(AVAILABLE_MODELS.keys())
    model_ids = list(AVAILABLE_MODELS.values())

    # Find default index
    default_model_id = settings.default_model
    default_idx = model_ids.index(default_model_id) if default_model_id in model_ids else 0

    selected_model_name = st.selectbox(
        "LLM Model",
        options=model_names,
        index=default_idx,
        key="model_select",
        help="Choose which model to use for story generation. Different models have different capabilities and costs.",
    )
    selected_model_id = AVAILABLE_MODELS[selected_model_name]
    st.session_state.selected_model = selected_model_id

    # Show model ID badge
    st.markdown(f'<span class="badge badge-purple">📡 {selected_model_id}</span>', unsafe_allow_html=True)

    # Custom model option
    custom_model = st.text_input(
        "Or enter custom model ID",
        value="",
        placeholder="e.g. mistralai/mistral-large",
        help="Enter any model ID from https://openrouter.ai/models",
        key="custom_model_input",
    )
    if custom_model.strip():
        selected_model_id = custom_model.strip()
        st.session_state.selected_model = selected_model_id
        st.markdown(f'<span class="badge badge-blue">🔧 Custom: {selected_model_id}</span>', unsafe_allow_html=True)

    st.divider()

    # ── Document Management ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">📚 Superman App Docs</div>', unsafe_allow_html=True)

    docs_folder = Path("docs")
    preloaded_files = list(docs_folder.glob("*")) if docs_folder.exists() else []
    preloaded_count = sum(1 for f in preloaded_files if not f.name.startswith("."))

    if preloaded_count > 0:
        st.markdown(f'<span class="badge badge-green">📁 {preloaded_count} pre-loaded doc(s)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-amber">📂 No pre-loaded docs (drop files in /docs)</span>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload additional docs",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, PPTX, TXT or MD files. These will be merged with pre-loaded docs.",
        key="doc_uploader",
        label_visibility="visible",
    )

    build_btn = st.button(
        "🔄 Build Knowledge Base",
        type="secondary",
        use_container_width=True,
        key="build_kb_btn",
        help="Index all docs into the vector store. Required before generating context-aware stories.",
    )

    if build_btn:
        if not api_key_input:
            st.error("⚠️ Please enter your OpenRouter API key first.")
        else:
            with st.spinner("Indexing documents..."):
                try:
                    all_docs = []
                    if preloaded_count > 0:
                        preloaded = load_docs_from_folder(str(docs_folder))
                        all_docs.extend(preloaded)

                    if uploaded_files:
                        # Reset file pointers
                        for f in uploaded_files:
                            f.seek(0)
                        uploaded = load_docs_from_uploads(uploaded_files)
                        all_docs.extend(uploaded)

                    if not all_docs:
                        st.warning("⚠️ No documents found. Upload files or add them to the /docs folder.")
                    else:
                        embeddings = get_embeddings(api_key=api_key_input)
                        vs = build_vectorstore(all_docs, embeddings)
                        retriever = get_retriever(vs)

                        st.session_state.vectorstore = vs
                        st.session_state.retriever = retriever
                        st.session_state.docs_loaded = True
                        st.session_state.docs_count = len(all_docs)

                        st.success(f"✅ Indexed {len(all_docs)} document chunks!")
                except Exception as e:
                    st.error(f"❌ Failed to build knowledge base: {e}")

    if st.session_state.docs_loaded:
        st.markdown(f"""
        <div style="margin-top:0.5rem;">
            <span class="badge badge-green">✅ KB Ready — {st.session_state.docs_count} chunks</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Jira Configuration ─────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-title">🔗 Jira Configuration</div>', unsafe_allow_html=True)

    jira_url = st.text_input(
        "Jira Base URL",
        value=settings.jira_base_url or "",
        placeholder="https://yourorg.atlassian.net",
        key="jira_url_input",
        label_visibility="visible",
    )
    jira_email = st.text_input(
        "Jira Email",
        value=settings.jira_email or "",
        placeholder="you@company.com",
        key="jira_email_input",
    )
    jira_token = st.text_input(
        "Jira API Token",
        type="password",
        value=settings.jira_api_token or "",
        placeholder="Your Jira API token",
        key="jira_token_input",
        help="Generate at: https://id.atlassian.com/manage-profile/security/api-tokens",
    )
    jira_project = st.text_input(
        "Project Key",
        value=settings.jira_project_key or "",
        placeholder="SUP",
        key="jira_project_input",
        max_chars=20,
    )
    jira_issue_type = st.selectbox(
        "Issue Type",
        options=["Story", "Task", "Bug", "Epic", "Sub-task"],
        index=0,
        key="jira_issue_type",
    )

    test_jira_btn = st.button("🔌 Test Jira Connection", use_container_width=True, key="test_jira_btn")
    if test_jira_btn:
        if jira_url and jira_email and jira_token:
            with st.spinner("Testing connection..."):
                client = JiraClient(jira_url, jira_email, jira_token)
                ok, msg = client.test_connection()
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        else:
            st.warning("⚠️ Fill in all Jira fields first.")

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem; color:#64748b; text-align:center; padding-bottom:1rem;">
        Powered by LangGraph · OpenRouter · FAISS · Streamlit<br>
        Superman Story Agent v2.0
    </div>
    """, unsafe_allow_html=True)


# ── Main Content Area ─────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🦸 Superman Story Agent</h1>
    <p class="hero-subtitle">
        Generate production-ready Jira User Stories in BDD format — powered by OpenRouter &amp; grounded in your app documentation
    </p>
    <div style="display:flex; gap:10px; margin-top:1rem; flex-wrap:wrap;">
        <span class="badge" style="background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3);">🧠 LangGraph Agent</span>
        <span class="badge" style="background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3);">📡 OpenRouter</span>
        <span class="badge" style="background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3);">📚 RAG Context</span>
        <span class="badge" style="background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3);">✅ BDD Format</span>
        <span class="badge" style="background:rgba(255,255,255,0.15); color:white; border:1px solid rgba(255,255,255,0.3);">🚀 Auto Jira Push</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    docs_status = "✅ Ready" if st.session_state.docs_loaded else "⚠️ Not Built"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{"📚" if st.session_state.docs_loaded else "📂"}</div>
        <div class="metric-label">Knowledge Base</div>
        <div style="font-size:0.7rem; color:{'#10b981' if st.session_state.docs_loaded else '#f59e0b'}; margin-top:4px;">{docs_status}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(st.session_state.generated_stories)}</div>
        <div class="metric-label">Stories Generated</div>
    </div>""", unsafe_allow_html=True)
with col3:
    pushed = sum(1 for r in st.session_state.jira_results.values() if r.get("success"))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pushed}</div>
        <div class="metric-label">Pushed to Jira</div>
    </div>""", unsafe_allow_html=True)
with col4:
    score = st.session_state.review.get("quality_score", 0) if st.session_state.review else 0
    score_color = "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="background: none; -webkit-text-fill-color: {score_color};">{score if score else "—"}</div>
        <div class="metric-label">AI Quality Score</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Story Generation Form ──────────────────────────────────────────────────
st.markdown("### 📝 Describe Your Feature")

with st.container():
    feature_input = st.text_area(
        "Feature / Requirement Description",
        height=130,
        placeholder=(
            "Example: Add a real-time notification system to the Superman app so that users receive "
            "instant alerts when their case status changes, when a new message arrives from a hero, "
            "or when an SLA deadline is approaching."
        ),
        key="feature_input",
        label_visibility="collapsed",
    )

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        num_stories = st.slider(
            "Number of stories to generate",
            min_value=1,
            max_value=5,
            value=1,
            key="num_stories_slider",
            help="How many Jira user stories should the agent generate?",
        )
    with col_b:
        has_stories = bool(st.session_state.generated_stories)
        generate_btn = st.button(
            "✅ Stories Generated" if has_stories else "⚡ Generate Stories",
            type="secondary" if has_stories else "primary",
            use_container_width=True,
            key="generate_btn",
            disabled=has_stories,
        )
    with col_c:
        clear_btn = st.button(
            "🗑️ Clear",
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

# ── Trigger Generation ─────────────────────────────────────────────────────
if generate_btn:
    if not api_key_input:
        st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
    elif not feature_input.strip():
        st.warning("⚠️ Please describe the feature or requirement.")
    else:
        # Show thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(f"""
        <div class="thinking-indicator">
            <div class="dot-pulse"><span></span><span></span><span></span></div>
            <div>
                <div style="font-weight:600; color:#a78bfa;">🧠 Agent is thinking... (via {selected_model_id})</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:2px;">
                    Step 1: Retrieving context from Superman docs &rarr; Step 2: Generating BDD stories &rarr; Step 3: Self-review
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        progress = st.progress(0, text="Initialising agent...")

        try:
            # Step 1
            progress.progress(10, text="🔍 Retrieving context from Superman docs...")
            time.sleep(0.3)

            # Prepare initial state
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

            progress.progress(30, text=f"✍️ Generating BDD stories via {selected_model_id}...")

            # Run the agent
            result = agent_app.invoke(initial_state)

            progress.progress(80, text="🔍 Running quality review...")
            time.sleep(0.3)
            progress.progress(100, text="✅ Done!")
            time.sleep(0.5)
            progress.empty()
            thinking_placeholder.empty()

            if result.get("error"):
                st.session_state.generation_error = result["error"]
                st.error(f"❌ Generation failed: {result['error']}")
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
            st.error(f"❌ Unexpected error: {e}")


# ── Display Review Summary ─────────────────────────────────────────────────
if st.session_state.review:
    review = st.session_state.review
    score = review.get("quality_score", 0)
    issues = review.get("issues", [])
    suggestions = review.get("suggestions", [])
    approved = review.get("approved", False)

    score_color = "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    status_badge = "badge-green" if approved else "badge-red"
    status_text = "AUTO-APPROVED" if approved else "NEEDS REVIEW"

    with st.expander(f"🔍 AI Quality Review — Score: {score}/100  |  Status: {status_text}", expanded=not approved):
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem;">
                <div style="font-size:3rem; font-weight:800; color:{score_color};">{score}</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Quality Score</div>
                <div style="margin-top:0.5rem;"><span class="badge {status_badge}">{status_text}</span></div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            if issues:
                st.markdown("**⚠️ Issues found:**")
                for issue in issues:
                    st.markdown(f"- {issue}")
            if suggestions:
                st.markdown("**💡 Suggestions:**")
                for s in suggestions:
                    st.markdown(f"- {s}")
            if not issues and not suggestions:
                st.markdown("✅ **All checks passed!** Stories meet BDD quality standards.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Display Generated Stories ──────────────────────────────────────────────
if st.session_state.generated_stories:
    stories = st.session_state.generated_stories
    total = len(stories)

    st.markdown(f"### 📋 Generated Stories ({total})")

    # Batch approve button
    all_pushed = all(
        st.session_state.jira_results.get(str(s["index"]), {}).get("success")
        for s in stories
    )
    if not all_pushed and jira_url and jira_email and jira_token and jira_project:
        if st.button(
            f"🚀 Approve & Push ALL {total} Stories to Jira",
            type="primary",
            key="push_all_btn",
            use_container_width=True,
        ):
            with st.spinner("Pushing all stories to Jira..."):
                for story in stories:
                    idx = str(story["index"])
                    if not st.session_state.jira_results.get(idx, {}).get("success"):
                        edited_md = st.session_state.edited_stories.get(idx, story["markdown"])
                        story_to_push = {**story, "markdown": edited_md}
                        result = push_story_to_jira(
                            story=story_to_push,
                            project_key=jira_project,
                            issue_type=jira_issue_type,
                            jira_base_url=jira_url,
                            jira_email=jira_email,
                            jira_api_token=jira_token,
                        )
                        st.session_state.jira_results[idx] = result
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Individual Story Cards ─────────────────────────────────────────────
    for story in stories:
        idx = str(story["index"])
        jira_result = st.session_state.jira_results.get(idx, {})
        already_pushed = jira_result.get("success", False)

        priority_colors = {
            "High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981",
            "Critical": "#dc2626", "Lowest": "#6b7280",
        }
        priority = story.get("priority", "Medium")
        p_color = priority_colors.get(priority, "#f59e0b")

        # Card container
        st.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #1a1a2e, #12121f);
            border: 1px solid rgba(124,58,237,{0.5 if already_pushed else 0.2});
            border-radius: 16px;
            padding: 1.5rem 1.5rem 0.5rem;
            margin-bottom: 0.5rem;
            box-shadow: {'0 0 20px rgba(16,185,129,0.1)' if already_pushed else '0 4px 24px rgba(0,0,0,0.3)'};
        ">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; margin-bottom:1rem;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span class="badge badge-purple">STORY {story['index']}/{total}</span>
                    <span style="font-size:0.75rem; font-weight:600; color:{p_color};">● {priority}</span>
                    <span style="font-size:0.75rem; color:#64748b;">⚡ {story.get('story_points', '?')} pts</span>
                </div>
                <div>
                    {'<span class="badge badge-green">✅ Pushed to Jira: <a href="' + jira_result.get("url","#") + '" target="_blank" style="color:#10b981;">' + jira_result.get("key","") + '</a></span>' if already_pushed else ""}
                </div>
            </div>
            <div style="font-size:1.05rem; font-weight:700; color:#f1f5f9; margin-bottom:1rem;">{story.get('title','')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs: Preview | Edit
        tab_preview, tab_edit = st.tabs([f"📖 Preview — Story {story['index']}", f"✏️ Edit — Story {story['index']}"])

        with tab_preview:
            current_md = st.session_state.edited_stories.get(idx, story["markdown"])
            st.markdown(current_md)

            if jira_result.get("error") and not already_pushed:
                st.warning(f"⚠️ Jira note: {jira_result['error']}")

        with tab_edit:
            edited = st.text_area(
                f"Edit Story {story['index']} (Markdown)",
                value=st.session_state.edited_stories.get(idx, story["markdown"]),
                height=450,
                key=f"edit_story_{idx}",
                label_visibility="collapsed",
            )
            if edited != st.session_state.edited_stories.get(idx, story["markdown"]):
                st.session_state.edited_stories[idx] = edited

        # Action buttons
        col_dl, col_approve = st.columns([1, 1])
        with col_dl:
            current_md = st.session_state.edited_stories.get(idx, story["markdown"])
            st.download_button(
                label=f"⬇️ Download Story {story['index']}",
                data=current_md,
                file_name=f"story_{story['index']}_bdd.md",
                mime="text/markdown",
                key=f"dl_{idx}",
                use_container_width=True,
            )

        with col_approve:
            if already_pushed:
                st.markdown(f"""
                <div style="
                    background: rgba(16,185,129,0.1);
                    border: 1px solid rgba(16,185,129,0.3);
                    border-radius: 10px;
                    padding: 0.6rem 1rem;
                    text-align:center;
                ">
                    <a href="{jira_result.get('url','#')}" target="_blank" style="
                        color:#10b981; font-weight:700; text-decoration:none; font-family:'JetBrains Mono',monospace;
                    ">🔗 {jira_result.get('key','Pushed')} — View in Jira</a>
                </div>""", unsafe_allow_html=True)
            else:
                # Check if Jira is configured
                jira_configured = bool(jira_url and jira_email and jira_token and jira_project)
                push_btn = st.button(
                    f"✅ Approve & Push Story {story['index']} to Jira",
                    type="primary",
                    use_container_width=True,
                    key=f"push_{idx}",
                    disabled=not jira_configured,
                    help="Configure Jira settings in the sidebar to enable pushing." if not jira_configured else "Push this story to Jira.",
                )
                if push_btn:
                    with st.spinner(f"Pushing Story {story['index']} to Jira..."):
                        current_md = st.session_state.edited_stories.get(idx, story["markdown"])
                        story_to_push = {**story, "markdown": current_md}
                        result = push_story_to_jira(
                            story=story_to_push,
                            project_key=jira_project,
                            issue_type=jira_issue_type,
                            jira_base_url=jira_url,
                            jira_email=jira_email,
                            jira_api_token=jira_token,
                        )
                        st.session_state.jira_results[idx] = result
                        if result["success"]:
                            st.success(f"✅ Created {result['key']}! [View in Jira]({result['url']})")
                        else:
                            st.error(f"❌ Failed to push: {result['error']}")
                    st.rerun()

        st.divider()

# ── Empty state ────────────────────────────────────────────────────────────
elif not st.session_state.generation_error:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem;
        background: linear-gradient(145deg, #1a1a2e, #12121f);
        border: 1px dashed rgba(124,58,237,0.3);
        border-radius: 16px;
        margin-top: 1rem;
    ">
        <div style="font-size:3rem; margin-bottom:1rem;">🦸‍♂️</div>
        <div style="font-size:1.2rem; font-weight:700; color:#a78bfa; margin-bottom:0.5rem;">
            Ready to Generate
        </div>
        <div style="color:#64748b; max-width:500px; margin:0 auto; line-height:1.6;">
            Describe your feature above, set your OpenRouter API key in the sidebar,
            and optionally load your Superman app documentation for context-aware story generation.
        </div>
        <div style="margin-top:1.5rem; display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
            <span class="badge badge-purple">1️⃣ Set OpenRouter Key</span>
            <span class="badge badge-blue">2️⃣ Pick Model</span>
            <span class="badge badge-green">3️⃣ Load Docs (optional)</span>
            <span class="badge badge-amber">4️⃣ Describe Feature</span>
            <span class="badge badge-purple">5️⃣ Generate & Push to Jira</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
