import time
from ollama import Client as OllamaClient, ChatResponse

from config.config import settings
from config.logger import get_logger

logger = get_logger(__name__)

_ollama = OllamaClient(host=settings.ollama_base_url)

SYSTEM_PROMPT = """
You are Pensieve AI, a technical assistant for this codebase only.

STRICT RULES — no exceptions:
1. You may ONLY use information explicitly present in the ### Retrieved Context below.
2. If the context does not contain enough information to answer, you MUST respond with
   exactly: "I don't have enough context in the indexed knowledge base to answer this."
   Do NOT attempt to answer from general knowledge.
3. The codebase you are helping with is private and internal.
   Do NOT draw on any knowledge of public libraries, open-source projects,
   or anything outside the retrieved context.
4. Never say "typically", "usually", or "in most frameworks" —
   these are signals you are hallucinating from training data.

If you follow these rules, say only what the context supports.
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