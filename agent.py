"""
agent.py — LangGraph multi-node agent for Jira BDD Story Generation.

Uses OpenRouter as the LLM provider (OpenAI-compatible API).
All models (GPT-4o, Claude, Gemini, Llama, etc.) are accessed via a single API key.

Nodes:
  1. retrieve_context  — RAG over Superman app docs
  2. generate_stories  — BDD story generation with LLM
  3. review_stories    — LLM self-review / quality check
  4. push_to_jira      — Jira REST API push (conditional, triggered externally)
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from config import OPENROUTER_BASE_URL
from prompts import (
    STORY_SYSTEM_PROMPT,
    STORY_GENERATION_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_PROMPT,
)


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Inputs
    user_input: str
    num_stories: int
    model: str              # OpenRouter model ID e.g. "openai/gpt-4o"

    # RAG
    retriever: Optional[Any]  # LangChain retriever (not serializable — passed in memory)
    context: str

    # Outputs
    stories_markdown: str       # Full markdown output from LLM
    stories: List[Dict]         # Parsed list of individual story dicts
    review: Dict                # JSON review result
    error: str


def _get_api_key() -> str:
    """Read the OpenRouter API key from the environment (set by the UI layer)."""
    return os.environ.get("OPENROUTER_API_KEY", "")


# ── LLM factory (OpenRouter) ─────────────────────────────────────────────────

def _get_llm(api_key: str, model: str):
    """
    Create a ChatOpenAI instance pointing to OpenRouter.
    OpenRouter is fully OpenAI-compatible, so we just override the base_url.
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=0.4,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Superman Jira Story Agent",
        },
    )


# ── Node 1: Retrieve Context ──────────────────────────────────────────────────

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from the Superman app docs vector store."""
    from rag_engine import retrieve_context as _retrieve

    retriever = state.get("retriever")
    query = state.get("user_input", "")

    context = _retrieve(retriever, query)
    return {**state, "context": context, "error": ""}


# ── Node 2: Generate Stories ──────────────────────────────────────────────────

def generate_stories(state: AgentState) -> AgentState:
    """Generate BDD Jira user stories using the LLM via OpenRouter."""
    try:
        llm = _get_llm(_get_api_key(), state["model"])

        prompt_content = STORY_GENERATION_PROMPT.format(
            context=state.get("context", "No documentation context available."),
            user_input=state["user_input"],
            num_stories=state.get("num_stories", 1),
        )

        messages = [
            SystemMessage(content=STORY_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        response = llm.invoke(messages)
        stories_markdown = response.content

        # Parse individual stories from the markdown
        stories = _parse_stories(stories_markdown, state.get("num_stories", 1))

        return {
            **state,
            "stories_markdown": stories_markdown,
            "stories": stories,
            "error": "",
        }
    except Exception as e:
        return {**state, "stories_markdown": "", "stories": [], "error": str(e)}


def _parse_stories(markdown: str, num_stories: int) -> List[Dict]:
    """
    Split the combined markdown output into individual story dicts.
    Each story dict has: {title, markdown, story_points, priority, labels}
    """
    # Split on the story delimiter pattern
    # Stories are separated by "---\n## 🎫 Story N:"
    pattern = r'(?:^|\n)---\n## 🎫 Story \d+:'
    parts = re.split(pattern, markdown)

    # Remove empty parts
    parts = [p.strip() for p in parts if p.strip()]

    # If splitting didn't work cleanly, return the whole thing as one story
    if not parts or len(parts) == 0:
        return [{"index": 1, "title": "Generated Story", "markdown": markdown, "story_points": 3, "priority": "Medium", "labels": []}]

    stories = []
    for i, part in enumerate(parts):
        # Extract title (first line of the part, or after the "Story N:" header)
        title_match = re.match(r'^([^\n]+)', part)
        title = title_match.group(1).strip() if title_match else f"Story {i+1}"
        # Clean emoji from title
        title = re.sub(r'[^\w\s\-:.()/&]', '', title).strip()

        # Extract story points
        sp_match = re.search(r'\|\s*\*\*Story Points\*\*\s*\|\s*(\d+)', part)
        story_points = int(sp_match.group(1)) if sp_match else 3

        # Extract priority
        pri_match = re.search(r'\|\s*\*\*Priority\*\*\s*\|\s*(\w+)', part)
        priority = pri_match.group(1) if pri_match else "Medium"

        # Extract labels
        lab_match = re.search(r'\|\s*\*\*Labels\*\*\s*\|\s*([^\|]+)', part)
        labels = []
        if lab_match:
            labels = [l.strip() for l in lab_match.group(1).split(",") if l.strip()]

        # Reconstruct full story markdown with the header
        full_md = f"## 🎫 Story {i+1}: {title}\n\n{part}" if not part.startswith(title) else f"## 🎫 Story {i+1}: {part}"

        stories.append({
            "index": i + 1,
            "title": title if title else f"Story {i+1}",
            "markdown": full_md,
            "story_points": story_points,
            "priority": priority,
            "labels": labels,
        })

    return stories


# ── Node 3: Review Stories ────────────────────────────────────────────────────

def review_stories(state: AgentState) -> AgentState:
    """LLM self-review of the generated stories for quality assurance."""
    if state.get("error") or not state.get("stories_markdown"):
        return {**state, "review": {"quality_score": 0, "issues": [], "suggestions": [], "approved": False}}

    try:
        llm = _get_llm(_get_api_key(), state["model"])

        prompt_content = REVIEW_PROMPT.format(stories=state["stories_markdown"])

        messages = [
            SystemMessage(content=REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Extract JSON from the response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            review = json.loads(json_match.group())
        else:
            review = {"quality_score": 75, "issues": [], "suggestions": [], "approved": True}

        return {**state, "review": review, "error": ""}
    except Exception as e:
        # Non-fatal: if review fails, still show the stories
        return {**state, "review": {"quality_score": 70, "issues": [str(e)], "suggestions": [], "approved": True}}


# ── Build the LangGraph ───────────────────────────────────────────────────────

def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_stories", generate_stories)
    workflow.add_node("review_stories", review_stories)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_stories")
    workflow.add_edge("generate_stories", "review_stories")
    workflow.add_edge("review_stories", END)

    return workflow.compile()


# Compiled agent (imported by app.py)
agent_app = build_agent()


# ── Jira Push (called from app.py on user approval) ──────────────────────────

def push_story_to_jira(
    story: Dict,
    project_key: str,
    issue_type: str,
    jira_base_url: str,
    jira_email: str,
    jira_api_token: str,
) -> Dict:
    """
    Push a single approved story to Jira.
    Returns the result dict from JiraClient.create_issue().
    """
    from jira_client import JiraClient

    client = JiraClient(jira_base_url, jira_email, jira_api_token)

    # Build the summary from the story title
    summary = story.get("title", "BDD User Story")
    if len(summary) > 255:
        summary = summary[:252] + "..."

    result = client.create_issue(
        project_key=project_key,
        summary=summary,
        description_markdown=story.get("markdown", ""),
        issue_type=issue_type,
        story_points=story.get("story_points"),
        labels=story.get("labels", []),
        priority=story.get("priority"),
    )
    return result
