"""
config.py — Settings loader for the Jira BDD Story Generator.
Reads from .env file and environment variables.
"""
import os
from dotenv import load_dotenv

# Load .env if present (silently skips if not found)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# ── OpenRouter Constants ──────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Popular models available on OpenRouter (display_name → model_id)
AVAILABLE_MODELS = {
    "GPT-4o (OpenAI)": "openai/gpt-4o",
    "GPT-4o Mini (OpenAI)": "openai/gpt-4o-mini",
    "Claude Sonnet 4 (Anthropic)": "anthropic/claude-sonnet-4",
    "Claude Haiku 3.5 (Anthropic)": "anthropic/claude-3.5-haiku",
    "Gemini 2.5 Flash (Google)": "google/gemini-2.5-flash-preview",
    "Gemini 2.5 Pro (Google)": "google/gemini-2.5-pro-preview",
    "DeepSeek Chat V3 (DeepSeek)": "deepseek/deepseek-chat-v3-0324",
    "Llama 3.1 70B (Meta)": "meta-llama/llama-3.1-70b-instruct",
    "Llama 4 Maverick (Meta)": "meta-llama/llama-4-maverick",
    "Qwen3 32B (Alibaba)": "qwen/qwen3-32b",
}

# Default embedding model via OpenRouter (OpenAI embeddings proxied)
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def get_setting(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


class Settings:
    """Central settings class — reads from Streamlit secrets or environment variables."""

    # ── OpenRouter ────────────────────────────────────────────────────────────
    @property
    def openrouter_api_key(self) -> str:
        return get_setting("OPENROUTER_API_KEY", "")

    @property
    def default_model(self) -> str:
        return get_setting("OPENROUTER_MODEL", "openai/gpt-4o")

    # ── Jira ──────────────────────────────────────────────────────────────────
    @property
    def jira_base_url(self) -> str:
        return get_setting("JIRA_BASE_URL", "").rstrip("/")

    @property
    def jira_email(self) -> str:
        return get_setting("JIRA_EMAIL", "")

    @property
    def jira_api_token(self) -> str:
        return get_setting("JIRA_API_TOKEN", "")

    @property
    def jira_project_key(self) -> str:
        return get_setting("JIRA_PROJECT_KEY", "")

    @property
    def jira_issue_type(self) -> str:
        return get_setting("JIRA_ISSUE_TYPE", "Story")

    # ── Derived helpers ────────────────────────────────────────────────────────
    @property
    def has_api_key(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_jira(self) -> bool:
        return bool(self.jira_base_url and self.jira_email and self.jira_api_token)


settings = Settings()
