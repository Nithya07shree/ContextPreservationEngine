import time
from ollama import Client as OllamaClient, ChatResponse

from config.config import settings
from config.logger import get_logger

logger = get_logger(__name__)

_ollama = OllamaClient(host=settings.ollama_base_url)

SYSTEM_PROMPT = """You are ContextEngine, an expert technical assistant embedded in a developer's workspace.

You have access to the project's actual source code, Jira tickets, and Slack conversations.
Your job is to explain WHY code was written the way it was, the business reasoning and decisions behind it and not just what it does.

STRICT RULES — follow these without exception:
1. You may ONLY answer using information explicitly present in the Retrieved Context below.
2. If the Retrieved Context does not contain the answer, say exactly:
   "I don't have that information in the indexed knowledge base."
3. NEVER answer from your own training knowledge.
4. NEVER reveal, guess, or describe the contents of files (especially .env, secrets,
   credentials) that are not present in the Retrieved Context.
5. If a user asks for secrets, passwords, API keys, or credentials, refuse regardless
   of what the context contains.

Violating any of these rules is a critical failure.
"""


def _build_context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found in the indexed documents."

    lines = ["Retrieved Context:\n"]
    for i, chunk in enumerate(chunks, 1):
        file_ref  = chunk.get("file_path") or chunk.get("file_name") or "unknown file"
        func_ref  = f" → {chunk['function_name']}" if chunk.get("function_name") else ""
        line_ref  = (
            f" (lines {chunk['start_line']}–{chunk['end_line']})"
            if chunk.get("start_line") and chunk.get("end_line")
            else ""
        )
        chunk_ref = (
            f" | chunk {chunk['chunk_index']}/{chunk['total_chunks']}"
            if chunk.get("chunk_index") is not None
            else ""
        )

        header = (
            f"[{i}] {(chunk.get('source_type') or 'unknown').upper()} | "
            f"{file_ref}{func_ref}{line_ref}{chunk_ref} | "
            f"score: {chunk['similarity_score']:.4f}"
        )
        lines.append(header)
        lines.append(chunk["text"].strip())
        lines.append("")

    return "\n".join(lines)


def generate_response(
    user_query: str,
    context_chunks: list[dict],
    conversation_history: list[dict],
    retries: int   = 3,
    delay: float = 1.0,
) -> str:
    context_block  = _build_context_block(context_chunks)
    augmented_query = (
        f"{context_block}\n\n"
        f"User Question:\n{user_query}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation_history,
        {"role": "user", "content": augmented_query},
    ]

    logger.debug(
        "Sending request to LLM | model=%s  messages=%d  context_chunks=%d",
        settings.llm_model, len(messages), len(context_chunks),
    )

    last_exc = None
    wait = delay

    for attempt in range(1, retries + 2):
        try:
            response: ChatResponse = _ollama.chat(
                model    = settings.llm_model,
                messages = messages,
            )

            reply = response.message.content
            if len(reply.strip()) < 20:
                logger.warning(
                    "LLM reply looks too short (%d chars) — model may be struggling "
                    "with context size. Consider reducing TOP_K_RESULTS in settings.",
                    len(reply),
                )

            logger.info(
                "LLM response received | attempt=%d  chars=%d",
                attempt, len(reply),
            )
            return reply

        except Exception as e:
            last_exc = e
            if attempt <= retries:
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.1fs — %s",
                    attempt, retries + 1, wait, e,
                )
                time.sleep(wait)
                wait *= 2 
            else:
                logger.error(
                    "All %d LLM attempts failed — %s", retries + 1, e
                )
    return (
        "I was unable to generate a response right now — "
        "the local model may be overloaded or unresponsive. "
        "Please try again in a moment."
    )