"""
jira_client.py — Jira REST API v3 wrapper.

Creates Jira issues from generated BDD stories using the Atlassian REST API.
No external Jira SDK required — uses the standard `requests` library.
"""
import json
import re
from typing import Optional
import requests
from requests.auth import HTTPBasicAuth


class JiraClient:
    """Thin wrapper around the Jira REST API v3."""

    def __init__(self, base_url: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPBasicAuth(email.strip(), api_token.strip())
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        response = requests.post(
            url,
            headers=self.headers,
            auth=self.auth,
            data=json.dumps(payload),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _get(self, endpoint: str) -> dict:
        url = f"{self.base_url}/rest/api/3/{endpoint}"
        response = requests.get(
            url,
            headers=self.headers,
            auth=self.auth,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def test_connection(self) -> tuple[bool, str]:
        """Test the Jira connection and return (success, message)."""
        try:
            data = self._get("myself")
            return True, f"Connected as: {data.get('displayName', 'Unknown')} ({data.get('emailAddress', '')})"
        except requests.exceptions.HTTPError as e:
            return False, f"HTTP Error {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return False, str(e)

    def get_projects(self) -> list:
        """Return list of accessible projects."""
        try:
            data = self._get("project")
            return [{"key": p["key"], "name": p["name"]} for p in data]
        except Exception:
            return []

    def markdown_to_adf(self, markdown_text: str) -> dict:
        """
        Convert Markdown story text to Atlassian Document Format (ADF).
        Uses a simplified conversion — keeps Gherkin blocks as code blocks.
        """
        paragraphs = []
        lines = markdown_text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Heading
            if line.startswith("## ") or line.startswith("### ") or line.startswith("#### "):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("# ").strip()
                paragraphs.append({
                    "type": "heading",
                    "attrs": {"level": min(level, 6)},
                    "content": [{"type": "text", "text": text}]
                })
                i += 1

            # Gherkin code block
            elif line.startswith("```"):
                lang = line[3:].strip() or "text"
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                i += 1  # skip closing ```
                code_content = "\n".join(code_lines)
                paragraphs.append({
                    "type": "codeBlock",
                    "attrs": {"language": lang},
                    "content": [{"type": "text", "text": code_content}]
                })

            # Blockquote
            elif line.startswith("> "):
                text = line[2:].strip()
                # Strip markdown bold
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
                paragraphs.append({
                    "type": "blockquote",
                    "content": [{
                        "type": "paragraph",
                        "content": [{"type": "text", "text": text}]
                    }]
                })
                i += 1

            # Table row (skip table, convert to plain paragraph)
            elif line.startswith("|"):
                # Collect table lines and convert to a simple bullet list
                table_rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row = lines[i].strip()
                    if not re.match(r'^\|[\s\-|]+\|$', row):  # skip separator rows
                        cells = [c.strip() for c in row.split("|") if c.strip()]
                        if cells:
                            table_rows.append(cells)
                    i += 1
                # Render as bullet list
                if len(table_rows) > 1:  # skip header row
                    items = []
                    for row in table_rows[1:]:
                        if len(row) >= 2:
                            items.append({
                                "type": "listItem",
                                "content": [{
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": f"{row[0]}: {row[1]}"}]
                                }]
                            })
                    if items:
                        paragraphs.append({"type": "bulletList", "content": items})

            # Bullet / task item
            elif line.startswith("- ") or line.startswith("* "):
                # Collect consecutive list items
                items = []
                while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                    item_text = lines[i].strip()[2:].strip()
                    # Strip markdown bold/italic
                    item_text = re.sub(r'\*\*(.+?)\*\*', r'\1', item_text)
                    item_text = re.sub(r'\*(.+?)\*', r'\1', item_text)
                    item_text = re.sub(r'`(.+?)`', r'\1', item_text)
                    items.append({
                        "type": "listItem",
                        "content": [{
                            "type": "paragraph",
                            "content": [{"type": "text", "text": item_text}]
                        }]
                    })
                    i += 1
                paragraphs.append({"type": "bulletList", "content": items})

            # Empty line
            elif not line:
                i += 1

            # Regular paragraph
            else:
                # Strip markdown bold/italic for plain text
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
                text = re.sub(r'\*(.+?)\*', r'\1', text)
                text = re.sub(r'`(.+?)`', r'\1', text)
                text = re.sub(r'^#{1,6}\s*', '', text)  # strip any remaining headings
                if text:
                    paragraphs.append({
                        "type": "paragraph",
                        "content": [{"type": "text", "text": text}]
                    })
                i += 1

        return {
            "version": 1,
            "type": "doc",
            "content": paragraphs if paragraphs else [{
                "type": "paragraph",
                "content": [{"type": "text", "text": "See story details."}]
            }]
        }

    def create_issue(
        self,
        project_key: str,
        summary: str,
        description_markdown: str,
        issue_type: str = "Story",
        story_points: Optional[int] = None,
        labels: Optional[list] = None,
        priority: Optional[str] = None,
    ) -> dict:
        """
        Create a Jira issue and return a dict with {key, url, id}.

        Args:
            project_key: Jira project key (e.g. 'SUP')
            summary: Issue title / summary
            description_markdown: Full BDD story in Markdown
            issue_type: 'Story', 'Task', 'Bug', etc.
            story_points: Fibonacci estimate
            labels: List of label strings
            priority: 'Highest', 'High', 'Medium', 'Low', 'Lowest'

        Returns:
            {'key': 'SUP-123', 'url': 'https://...', 'id': '10001'}
        """
        description_adf = self.markdown_to_adf(description_markdown)

        fields: dict = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description_adf,
            "issuetype": {"name": issue_type},
        }

        if labels:
            fields["labels"] = labels

        if priority:
            fields["priority"] = {"name": priority}

        # Story points — field name varies by Jira config
        # Common field IDs: "story_points", "customfield_10016", "customfield_10028"
        if story_points is not None:
            fields["story_points"] = story_points  # may be overridden per instance

        payload = {"fields": fields}

        try:
            result = self._post("issue", payload)
            issue_key = result.get("key", "")
            issue_id = result.get("id", "")
            issue_url = f"{self.base_url}/browse/{issue_key}"
            return {
                "success": True,
                "key": issue_key,
                "id": issue_id,
                "url": issue_url,
                "error": None,
            }
        except requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response is not None else str(e)
            try:
                # Try to parse Jira's JSON error for cleaner display
                parsed_error = json.loads(error_body)
                error_msg = json.dumps(parsed_error.get("errors", parsed_error), indent=2)
                if not parsed_error.get("errors") and parsed_error.get("errorMessages"):
                    error_msg = json.dumps(parsed_error.get("errorMessages"), indent=2)
            except Exception:
                error_msg = error_body

            # Fallback: Jira often rejects custom fields, priorities, or labels.
            # If we fail, let's try a minimal payload.
            fallback_fields = {
                "project": {"key": project_key},
                "summary": summary,
                "description": description_adf,
                "issuetype": {"name": issue_type},
            }
            if fields != fallback_fields:
                try:
                    result = self._post("issue", {"fields": fallback_fields})
                    issue_key = result.get("key", "")
                    return {
                        "success": True,
                        "key": issue_key,
                        "id": result.get("id", ""),
                        "url": f"{self.base_url}/browse/{issue_key}",
                        "error": f"Created successfully, but some fields (Priority/Story Points) were ignored. Original Error: {error_msg}",
                    }
                except requests.exceptions.HTTPError as e2:
                    error_body2 = e2.response.text if e2.response is not None else str(e2)
                    try:
                        p2 = json.loads(error_body2)
                        em2 = json.dumps(p2.get("errors", p2.get("errorMessages", p2)), indent=2)
                    except Exception:
                        em2 = error_body2
                    return {"success": False, "key": "", "id": "", "url": "", "error": f"Jira rejected the request. Reason:\n{em2}"}
            
            return {"success": False, "key": "", "id": "", "url": "", "error": f"Jira rejected the request. Reason:\n{error_msg}"}
        except Exception as e:
            return {"success": False, "key": "", "id": "", "url": "", "error": str(e)}
