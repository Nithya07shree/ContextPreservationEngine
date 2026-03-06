"""
Cleaner module — responsible for normalizing raw text from various sources before chunking.

Code: Code cleaning functions preserve structure of the code files and only remove trailing whitespaces, blank lines and normalize line endings.
Slack: Slack messages are cleaned with the help of regular expressions and stripped of any trailing whitespaces.
Jira: Normalizes format of the Jira file so that the chunker always recieves the expected CSV columns regardless of how the jira export looks like.
"""

import re
import json
import csv
import io
from pathlib import Path
from typing import Any

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".json": "json",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".cfg": "ini",
    ".ini": "ini",
    ".csv": "csv",
}

def detect_language(file_path: str | Path) -> str:
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "plaintext")

def clean_code(raw: str) -> str:
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.rstrip() for line in lines]
    cleaned = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()

# Slack cleaner

def clean_slack_message(text: str) -> str:
    """
    Normalize a single Slack message body.
    - Expand <@USERID> to @user
    - Expand <#CHANNELID|name> to #name
    - Remove <http://...> link wrappers, keep URL or label
    """
    if not text:
        return ""
    text = re.sub(r"<@([A-Z0-9]+)>", "@user", text)
    text = re.sub(r"<#[A-Z0-9]+\|([^>]+)>", r"#\1", text)
    text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 (\1)", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_slack_export(file_path: str | Path) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    messages = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                messages.extend(value)
    return messages

# Jira cleaner
def clean_jira_text(value: str) -> str:
    # To clean the value of a single field
    if not value:
        return ""
    value = value.replace("\\n", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _extract_jira_comments(row: dict[str, str]) -> list[str]:
    comment_keys = sorted(
        [k for k in row.keys() if re.match(r"Comment(\[\d+\])?$", k.strip(), re.IGNORECASE)],
        key=lambda k: int(m.group(1)) if (m := re.search(r"\[(\d+)\]", k)) else -1
    )

    comments = []
    for key in comment_keys:
        raw = row.get(key, "").strip()
        if not raw:
            continue
        # If comments are separated using a semi-colon.
        if re.search(r"\s;\s", raw):
            for entry in re.split(r"\s;\s", raw):
                entry = entry.strip()
                if entry:
                    comments.append(entry)
        else:
            # Only one proper column present
            comments.append(raw)

    return comments


def _get_field(row: dict[str, str], *candidates: str) -> str:
    row_normalized = {k.strip().lower(): v for k, v in row.items()}
    for candidate in candidates:
        val = row_normalized.get(candidate.strip().lower(), "").strip()
        if val:
            return val
    return ""


def load_jira_csv(file_path: str | Path) -> list[dict[str, str]]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in reader:
        cleaned_row = {k: clean_jira_text(v) for k, v in row.items() if k}
        rows.append(cleaned_row)
    return rows


def build_jira_content(row: dict[str, str]) -> str:
    """
    Build a single readable text blob from a Jira CSV row.
    Order: Summary -> Type/Status/Priority -> Description -> Comments.
    """
    parts = []

    summary = _get_field(row, "Summary", "Title", "Name", "Subject")
    if summary:
        parts.append(f"Summary: {summary}")

    issue_type = _get_field(row, "Issue Type", "IssueType", "Type")
    status = _get_field(row, "Status", "State")
    priority = _get_field(row, "Priority")
    if any([issue_type, status, priority]):
        meta = " | ".join(filter(None, [issue_type, status, priority]))
        parts.append(f"Type/Status/Priority: {meta}")

    description = _get_field(row, "Description", "Body", "Details", "Content")
    if description:
        parts.append(f"Description:\n{description}")

    comments = _extract_jira_comments(row)
    if comments:
        parts.append("Comments:\n" + "\n---\n".join(comments))

    # Safety fallback: if no known fields matched at all, all fields are preserved.
    if not parts:
        fallback = " | ".join(f"{k}: {v}" for k, v in row.items() if v.strip())
        if fallback:
            parts.append(fallback)

    return "\n\n".join(parts)