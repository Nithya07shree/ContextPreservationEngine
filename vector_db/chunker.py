import ast
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from cleaner import (
    clean_code,
    clean_slack_message,
    detect_language,
    build_jira_content,
)

MAX_CODE_LINES = 150     
MAX_SLACK_TOKENS = 512    
MAX_JIRA_TOKENS = 1000   
AVG_CHARS_PER_TOKEN = 4    


def make_chunk_id(*parts: str) -> str:
    raw = "|".join(str(p) for p in parts)
    hash_hex = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hash_hex[:32]))

def _parse_with_py2_fallback(source: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        pass

    patched = source
    patched = re.sub(r'^(\s*)print (?!\()(.+)$', r'\1print(\2)', patched, flags=re.MULTILINE)
    patched = re.sub(r'except\s+(\w+(?:\.\w+)*)\s*,\s*(\w+)\s*:', r'except \1 as \2:', patched)
    patched = patched.replace("basestring", "str")
    patched = re.sub(r'\bunicode\(', 'str(', patched)

    try:
        return ast.parse(patched)
    except SyntaxError:
        return None  # literally cannot be parsed. 


def _extract_python_functions(source: str, file_name: str) -> list[dict[str, Any]]:
    lines = source.split("\n")
    tree = _parse_with_py2_fallback(source)
    if tree is None:
        return [{
            "function_name": "__module__",
            "start_line": 1,
            "end_line": len(lines),
            "lines": lines,
        }]

    functions = []

    def visit(node, class_name=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{class_name}.{node.name}" if class_name else node.name
            start = node.lineno
            end = node.end_lineno
            functions.append({
                "function_name": name,
                "start_line": start,
                "end_line": end,
                "lines": lines[start - 1:end],
            })
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                visit(child, class_name=node.name)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    for node in ast.iter_child_nodes(tree):
        visit(node)
    if not functions:
        functions = [{
            "function_name": "__module__",
            "start_line": 1,
            "end_line": len(lines),
            "lines": lines,
        }]

    return functions


def _split_oversized_function(func: dict[str, Any], file_name: str) -> list[dict[str, Any]]:
    lines = func["lines"]
    if len(lines) <= MAX_CODE_LINES:
        return [func]

    sub_chunks = []
    start = 0
    while start < len(lines):
        end = min(start + MAX_CODE_LINES, len(lines))
        if end < len(lines):
            for i in range(end, max(start + MAX_CODE_LINES // 2, start + 1), -1):
                if lines[i - 1].strip() == "":
                    end = i
                    break
        sub_chunks.append({
            "function_name": func["function_name"],
            "start_line": func["start_line"] + start,
            "end_line": func["start_line"] + end - 1,
            "lines": lines[start:end],
        })
        start = end

    return sub_chunks


def _extract_functions_by_regex(source, file_name, pattern, language):
    lines = source.split('\n')
    functions = []

    for match in pattern.finditer(source):
        groups = match.groupdict()
        func_name = groups.get('name') or groups.get('ctor')
        if not func_name:
            continue

        start_line = source[:match.start()].count('\n') + 1

        brace_pos = source.find('{', match.start())
        if brace_pos == -1:
            continue

        depth = 0
        end_pos = brace_pos
        for i, ch in enumerate(source[brace_pos:], brace_pos):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        end_line = source[:end_pos].count('\n') + 1
        functions.append({
            'function_name': func_name,
            'start_line': start_line,
            'end_line': end_line,
            'lines': lines[start_line - 1:end_line],
        })

    if not functions:
        functions = [{
            'function_name': '__module__',
            'start_line': 1,
            'end_line': len(lines),
            'lines': lines,
        }]

    return functions


_PHP_FUNC_PATTERN = re.compile(
    "(?:(?:public|private|protected|static|abstract|final)\\s+)*"
    "function\\s+(?P<name>[a-zA-Z_\\x7f-\\xff][a-zA-Z0-9_\\x7f-\\xff]*)\\s*\\(",
    re.MULTILINE,
)

_JAVA_METHOD_PATTERN = re.compile(
    "(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)\\s+)+"
    "(?:"
        "(?:[a-zA-Z_$][a-zA-Z0-9_$<>\\[\\]]*\\s+)(?P<name>[a-zA-Z_$][a-zA-Z0-9_$]*)"
        "|(?P<ctor>[A-Z][a-zA-Z0-9_$]*)"
    ")\\s*\\([^)]*\\)\\s*"
    "(?:throws\\s+[a-zA-Z_$][a-zA-Z0-9_$,\\s]*\\s*)?",
    re.MULTILINE,
)

_JS_FUNC_PATTERN = re.compile(
    "(?:export\\s+)?(?:default\\s+)?(?:async\\s+)?function\\s*\\*?\\s*"
    "(?P<name>[a-zA-Z_$][a-zA-Z0-9_$]*)\\s*\\(",
    re.MULTILINE,
)


def chunk_code_file(file_path: str | Path) -> list[dict[str, Any]]:

    file_path = Path(file_path)
    language = detect_language(file_path)
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    source = clean_code(raw)
    file_name = file_path.name
    lines = source.split("\n")

    if language == "python":
        functions = _extract_python_functions(source, file_name)
    elif language == "php":
        functions = _extract_functions_by_regex(source, file_name, _PHP_FUNC_PATTERN, language)
    elif language == "java":
        functions = _extract_functions_by_regex(source, file_name, _JAVA_METHOD_PATTERN, language)
    elif language in ("javascript", "typescript"):
        functions = _extract_functions_by_regex(source, file_name, _JS_FUNC_PATTERN, language)
    elif language in ("smarty", "html", "css", "xml", "sql", "bash"):
        functions = []
        for i in range(0, len(lines), MAX_CODE_LINES):
            chunk_lines = lines[i:i + MAX_CODE_LINES]
            functions.append({
                "function_name": f"block_{i // MAX_CODE_LINES}",
                "start_line": i + 1,
                "end_line": i + len(chunk_lines),
                "lines": chunk_lines,
            })
    else:
        # All other languages: line-window fallback
        functions = []
        for i in range(0, len(lines), MAX_CODE_LINES):
            chunk_lines = lines[i:i + MAX_CODE_LINES]
            functions.append({
                "function_name": f"block_{i // MAX_CODE_LINES}",
                "start_line": i + 1,
                "end_line": i + len(chunk_lines),
                "lines": chunk_lines,
            })

    chunks = []
    for func in functions:
        sub_funcs = _split_oversized_function(func, file_name)
        total = len(sub_funcs)
        for idx, sub in enumerate(sub_funcs):
            content = "\n".join(sub["lines"])
            chunk_id = make_chunk_id(file_name, sub["function_name"], str(sub["start_line"]), str(idx))
            chunks.append({
                "chunk_id": chunk_id,
                "content": content,
                "metadata": {
                    "source_type": "code",
                    "file_name": file_name,
                    "file_path": str(file_path),
                    "function_name": sub["function_name"],
                    "language": language,
                    "start_line": sub["start_line"],
                    "end_line": sub["end_line"],
                    "chunk_index": idx,
                    "total_chunks": total,
                },
            })

    return chunks


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // AVG_CHARS_PER_TOKEN)


def _build_thread_content(messages: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    parts = []
    authors = []
    timestamps = []

    for msg in messages:
        user = msg.get("user", msg.get("username", "unknown"))
        ts = msg.get("ts", "")
        text = clean_slack_message(msg.get("text", ""))
        if not text:
            continue
        parts.append(f"[{user} @ {ts}]: {text}")
        if user not in authors:
            authors.append(user)
        if ts:
            timestamps.append(ts)

    content = "\n".join(parts)
    earliest_ts = min(timestamps) if timestamps else ""
    return content, authors, earliest_ts


def _split_oversized_thread(
    messages: list[dict[str, Any]],
    channel_name: str,
    file_path: str,
) -> list[dict[str, Any]]:
    sub_chunks = []
    current_batch: list[dict[str, Any]] = []
    current_tokens = 0

    def flush(batch, idx, total_hint=None):
        if not batch:
            return
        content, authors, ts = _build_thread_content(batch)
        thread_ts = batch[0].get("thread_ts", batch[0].get("ts", ""))
        chunk_id = make_chunk_id(channel_name, thread_ts, str(idx))
        sub_chunks.append({
            "chunk_id": chunk_id,
            "content": content,
            "metadata": {
                "source_type": "slack",
                "file_name": Path(file_path).name,
                "file_path": file_path,
                "channel_name": channel_name,
                "timestamp": ts,
                "authors": ", ".join(authors),
                "thread_ts": thread_ts,
                "chunk_index": idx,
                "total_chunks": -1,  # patched after all sub-chunks known
            },
        })

    idx = 0
    for msg in messages:
        text = clean_slack_message(msg.get("text", ""))
        msg_tokens = _estimate_tokens(text)
        if current_tokens + msg_tokens > MAX_SLACK_TOKENS and current_batch:
            flush(current_batch, idx)
            idx += 1
            current_batch = []
            current_tokens = 0
        current_batch.append(msg)
        current_tokens += msg_tokens

    flush(current_batch, idx)
    total = len(sub_chunks)
    for chunk in sub_chunks:
        chunk["metadata"]["total_chunks"] = total

    return sub_chunks


def chunk_slack_export(file_path: str | Path, channel_name: str = "") -> list[dict[str, Any]]:
    from cleaner import load_slack_export
    file_path = Path(file_path)

    if not channel_name:
        channel_name = file_path.parent.name or file_path.stem

    messages = load_slack_export(file_path)

    threads: dict[str, list[dict[str, Any]]] = {}
    standalone: list[dict[str, Any]] = []

    for msg in messages:
        thread_ts = msg.get("thread_ts")
        ts = msg.get("ts", "")
        if thread_ts:
            threads.setdefault(thread_ts, []).append(msg)
        else:
            standalone.append(msg)

    chunks = []

    # Process threads
    for thread_ts, thread_messages in threads.items():
        thread_messages.sort(key=lambda m: float(m.get("ts", 0)))
        content, authors, ts = _build_thread_content(thread_messages)

        if _estimate_tokens(content) > MAX_SLACK_TOKENS:
            chunks.extend(_split_oversized_thread(thread_messages, channel_name, str(file_path)))
        else:
            chunk_id = make_chunk_id(channel_name, thread_ts, "0")
            chunks.append({
                "chunk_id": chunk_id,
                "content": content,
                "metadata": {
                    "source_type": "slack",
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "channel_name": channel_name,
                    "timestamp": ts,
                    "authors": ", ".join(authors),
                    "thread_ts": thread_ts,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            })

    for msg in standalone:
        text = clean_slack_message(msg.get("text", ""))
        if not text:
            continue
        ts = msg.get("ts", "")
        user = msg.get("user", msg.get("username", "unknown"))
        chunk_id = make_chunk_id(channel_name, ts, "standalone")
        chunks.append({
            "chunk_id": chunk_id,
            "content": f"[{user} @ {ts}]: {text}",
            "metadata": {
                "source_type": "slack",
                "file_name": file_path.name,
                "file_path": str(file_path),
                "channel_name": channel_name,
                "timestamp": ts,
                "authors": user,
                "thread_ts": ts,
                "chunk_index": 0,
                "total_chunks": 1,
            },
        })

    return chunks

def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * AVG_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated]"


def chunk_jira_csv(file_path: str | Path) -> list[dict[str, Any]]:

    from cleaner import load_jira_csv
    file_path = Path(file_path)
    rows = load_jira_csv(file_path)
    chunks = []

    TICKET_ID_CANDIDATES = ["Issue key", "Issue Key", "Key", "IssueKey", "Issue id", "Issue Id", "ID"]

    def resolve_ticket_id(row: dict) -> str:

        row_normalized = {k.strip().lower(): v for k, v in row.items()}
        for candidate in TICKET_ID_CANDIDATES:
            val = row_normalized.get(candidate.strip().lower(), "").strip()
            if val:
                return val
        return ""

    if rows:
        print(f"[chunker] Jira CSV loaded: {len(rows)} rows")
        print(f"[chunker] Columns detected: {list(rows[0].keys())}")
        sample_id = resolve_ticket_id(rows[0])
        if not sample_id:
            print(f"[chunker] WARNING: Could not find ticket ID column.")
            print(f"[chunker] Column repr (check for hidden chars): {[repr(k) for k in rows[0].keys()]}")
        else:
            print(f"[chunker] Ticket ID column resolved. First ticket: {sample_id}")

    for row in rows:
        ticket_id = resolve_ticket_id(row)
        if not ticket_id:
            continue

        content = build_jira_content(row)
        content = _truncate_to_token_budget(content, MAX_JIRA_TOKENS)

        chunk_id = make_chunk_id(file_path.name, ticket_id)
        chunks.append({
            "chunk_id": chunk_id,
            "content": content,
            "metadata": {
                "source_type": "jira",
                "file_name": file_path.name,
                "file_path": str(file_path),
                "ticket_id": ticket_id,
                "created": row.get("Created", ""),
                "updated": row.get("Updated", ""),
                "assignee": row.get("Assignee", ""),
                "reporter": row.get("Reporter", ""),
                "status": row.get("Status", ""),
                "priority": row.get("Priority", ""),
                "chunk_index": 0,
                "total_chunks": 1,
            },
        })

    return chunks