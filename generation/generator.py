import time
from ollama import Client as OllamaClient, ChatResponse

from config.config import settings
from config.logger import get_logger

logger = get_logger(__name__)

_ollama = OllamaClient(host=settings.ollama_base_url)

SYSTEM_PROMPT = """You are ContextEngine, an expert technical assistant for a developer's workspace.

You are given a "Retrieved Context" section containing excerpts from the project's indexed documents (e.g. source code, docs). That is the ONLY information you have for this conversation.

Your job is to answer the user's question using that context: explain what the code does, why it was written that way, and any business or security decisions mentioned in the context.

STRICT RULES — follow these without exception:
1. You may ONLY use information explicitly present in the Retrieved Context provided in the user message.
2. When Retrieved Context is provided and contains relevant information, you MUST answer the question from it. Do not say you "cannot access" or "don't have access" — the context you received is what you have; use it to answer.
3. If the Retrieved Context is empty or says "No relevant context found", or the context does not contain the answer, say exactly:
   "I don't have that information in the indexed knowledge base."
4. NEVER answer from your own training knowledge. Never mention Jira, Slack, or other tools unless they appear in the Retrieved Context.
5. NEVER reveal, guess, or describe the contents of files (especially .env, secrets, credentials) that are not present in the Retrieved Context.
6. If a user asks for secrets, passwords, API keys, or credentials, refuse regardless of what the context contains.

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